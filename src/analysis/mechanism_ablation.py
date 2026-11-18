"""Reproducible paired summaries for GB and charging-policy ablations."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np
from scipy import stats

from analysis.statistics import QualityContext, nondominated, holm_adjust
from analysis.scenario_validation import bootstrap_mean, pooled_front
from experiments.validation_io import read_json, write_json, write_csv, write_text, verify_hashes
from experiments.mechanism_ablation import INSTANCES, VARIANTS, load_plan, search_path
from configuration.experiment import stable_seed

LABELS = {"full":"GBGA-MOVNS", "no_gb":"GA + Node-MOVNS", "energy_only":"Energy-feasibility-only", "soc_threshold":"SOC-threshold (20%)"}
SEARCH = ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "runtime_sec", "candidate_evaluations", "scenario_simulations")
OOS = ("HV", "IGD_plus", "PFS", "mean_cost", "mean_ra", "energy_failure_rate", "num_charge", "low_soc_driving_time_min", "empty_front_rate")
DISPLAY = {"HV":"HV", "IGD_plus":r"IGD$^{+}$", "PFS":"PFS", "best_cost":r"$\bar z_{\mathrm{cost}}$", "best_ra":r"$\bar z_{\mathrm{ra}}$",
    "runtime_sec":"Time (s)", "candidate_evaluations":r"$N_{\mathrm{eval}}$", "scenario_simulations":r"$N_{\mathrm{sim}}$",
    "mean_cost":r"$\mathbb{E}[C]$", "mean_ra":r"$\mathbb{E}[RA]$", "energy_failure_rate":r"Energy failures (\%)",
    "num_charge":"Charges", "low_soc_driving_time_min":"Low-SOC time (min)", "empty_front_rate":r"Empty fronts (\%)"}


def paired_statistics(rows):
    lookup = {(r["instance"],r["variant"],r["run_id"]):r for r in rows if r["stage"]=="oos"}
    tests = []
    for instance in INSTANCES:
        for variant in VARIANTS[1:]:
            for metric in ("HV","IGD_plus"):
                ids = [i for i in range(1,31) if lookup[instance,"full",i][metric] is not None and lookup[instance,variant,i][metric] is not None]
                raw = np.asarray([lookup[instance,"full",i][metric]-lookup[instance,variant,i][metric] for i in ids])
                if metric=="IGD_plus":
                    raw = -raw
                d = np.round(raw,12)
                nz = d[d!=0]
                seed = stable_seed(20261030,instance,len(tests),"paired_statistics")
                test = dict(instance=instance, comparison="full vs "+variant, metric=metric, paired_n=len(ids),
                            paired_run_ids=ids, positive_favors="full", p_raw=None, statistic=None, rank_biserial=None)
                if len(d)>=2:
                    if len(nz)==0:
                        test.update(p_raw=1.,statistic=0.,rank_biserial=0.,method="all_zero")
                    else:
                        tied = len(np.unique(abs(nz)))!=len(nz) or len(nz)!=len(d)
                        method = stats.PermutationMethod(n_resamples=100000,batch=512,rng=np.random.default_rng(seed)) if tied else "exact"
                        w = stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided",method=method)
                        ranks = stats.rankdata(abs(nz),method="average")
                        effect = float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
                        test.update(p_raw=float(w.pvalue),statistic=float(w.statistic),rank_biserial=effect,
                                    method="permutation_100000" if tied else "exact")
                else:
                    test["method"]="insufficient_pairs"
                test.update(zero_differences=int((d==0).sum()), **{"difference_"+k:v for k,v in bootstrap_mean(raw.tolist(),seed+1).items()})
                tests.append(test)
    assert len(tests)==36
    adjusted = holm_adjust([t["p_raw"] if t["p_raw"] is not None else 1. for t in tests])
    for t,p in zip(tests,adjusted):
        t["p_holm"] = float(p) if t["p_raw"] is not None else None
        t["significant"] = bool(t["p_raw"] is not None and p<.05)
        t["holm_family_size"]=36
    return tests


def retained_solutions(solutions):
    """Use one deterministic representative per duplicate objective point."""
    front = nondominated([[s["cost"]["mean"],s["ra"]["mean"]] for s in solutions])
    selected = []
    for point in front:
        selected.append(next(s for s in sorted(solutions,key=lambda x:x["solution_id"])
                             if np.max(np.abs(point-[s["cost"]["mean"],s["ra"]["mean"]]))<=1e-9))
    return front,selected


def quality(context, points):
    if not len(points):
        return {"HV":0., "IGD_plus":None, "PFS":0}
    if context is None or not context.reference_front:
        raise ValueError("Nonempty front without reference")
    return {k:v for k,v in context.evaluate(points).items() if k in ("HV","IGD_plus","PFS")}


def format_tables(groups, directory):
    index={(g["instance"],g["variant"],g["stage"],g["metric"]):g for g in groups}
    outputs=[]
    for name, variants, caption in (
        ("granular_ball_ablation",("full","no_gb"),"Ablation of granular-ball decomposition in initialization and MOVNS."),
        ("charging_policy_ablation",("full","energy_only","soc_threshold"),"Comparison of charging policies within GBGA-MOVNS.")):
        lines=[r"\begingroup",r"\small",r"\setlength{\tabcolsep}{3pt}",r"\renewcommand{\arraystretch}{1.05}",
               r"\begin{longtable}{llrrrrrrr}",r"\caption{"+caption+r"}\label{tab:"+name+r"}\\",
               r"\toprule",r"Variant & Metric & C103 & R103 & RC103 & C203 & R203 & RC203 & Avg. \\",
               r"\midrule",r"\endfirsthead",r"\toprule",r"Variant & Metric & C103 & R103 & RC103 & C203 & R203 & RC203 & Avg. \\",r"\midrule",r"\endhead"]
        md=["### "+caption,"","| Variant | Stage / metric | C103 | R103 | RC103 | C203 | R203 | RC203 | Avg. |","|---|---|"+"---:|"*7]
        for stage,metrics in (("search",SEARCH),("oos",OOS)):
            lines.append(r"\multicolumn{9}{l}{"+("Search: 1,200 s" if stage=="search" else "Independent validation: 2,000 scenarios")+r"}\\")
            for variant in variants:
                for m in metrics:
                    matrix={v:[index[i,v,stage,m]["mean"] for i in INSTANCES] for v in variants}
                    for values in matrix.values():
                        values.append(float(np.mean(values)) if all(x is not None for x in values) else None)
                    cells=[]; mcells=[]
                    for j,value in enumerate(matrix[variant]):
                        compare=[matrix[v][j] for v in variants if matrix[v][j] is not None]
                        best=(max(compare) if m in ("HV","PFS") else min(compare)) if compare else None
                        factor=100 if m in ("energy_failure_rate","empty_front_rate") else 1
                        cell="--" if value is None else f"{value*factor:.2f}"
                        bold=m in ("HV","IGD_plus","PFS","best_cost","best_ra","mean_cost","mean_ra") and value is not None and value==best
                        cells.append(r"\textbf{"+cell+"}" if bold else cell)
                        mcells.append("**"+cell+"**" if bold else cell)
                    label=LABELS[variant] if m==metrics[0] else ""
                    if name=="charging_policy_ablation" and variant=="full":
                        label="RA-based" if label else ""
                    lines.append(label.replace("%",r"\%")+" & "+DISPLAY[m]+" & "+" & ".join(cells)+r" \\")
                    md.append("| "+" | ".join([LABELS[variant],stage+" / "+m,*mcells])+" |")
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule",r"\end{longtable}",r"\endgroup"])
        write_text(directory/(name+".tex"),"\n".join(lines)+"\n")
        outputs.append("\n".join(md))
    return "\n\n".join(outputs)


def report(output):
    output=Path(output); plan=load_plan(output); directory=output/"reports"
    directory.mkdir(exist_ok=True)
    rows=[]; solution_rows=[]; contexts={}; hashes={}
    for instance in INSTANCES:
        records=[]; search_pool=[]; oos_pool=[]
        for task in plan["tasks"]:
            if task["instance"]!=instance:
                continue
            replay_dir=output/"replay"/task["task_key"]
            receipt=read_json(replay_dir/"receipt.json"); verify_hashes(receipt["files"])
            if task["variant"]=="full":
                verify_hashes(plan["baseline"][task["task_key"]]["files"])
            else:
                verify_hashes(read_json(output/"search"/task["task_key"]/"receipt.json")["files"])
            old=read_json(search_path(output,plan,task)); r=read_json(replay_dir/"run.json")
            records.append((task,old,r))
            hashes[str(replay_dir/"run.json")]=receipt["files"][str(replay_dir/"run.json")]
            search_pool.extend([s["cost"],s["ra"]] for s in old["archive"] if s["search_feasible"])
            oos_pool.extend([s["cost"]["mean"],s["ra"]["mean"]] for s in r["solutions"] if s["eligible"])
        assert len(records)==120
        bounds=np.asarray(search_pool+oos_pool,float).reshape(-1,2)
        ctx={}
        for stage,pool in (("search",search_pool),("oos",oos_pool)):
            if len(bounds):
                lo=bounds.min(axis=0); hi=np.maximum(bounds.max(axis=0),lo+1e-12)
                hi=np.where(bounds.max(axis=0)>lo,bounds.max(axis=0),lo+1.)
                ctx[stage]=QualityContext(lo.tolist(),hi.tolist(),pooled_front(pool).tolist(),source="four_configurations_common_bounds; "+stage)
            else:
                ctx[stage]=None
        contexts[instance]={k:asdict(v) if v else None for k,v in ctx.items()}
        for task,old,r in records:
            identity={k:task[k] for k in ("instance","variant","run_id")}
            source=[s for s in old["archive"] if s["search_feasible"]]
            points=[[s["cost"],s["ra"]] for s in source]
            row={**identity,"stage":"search",**quality(ctx["search"],points),
                 "best_cost":min((s["cost"] for s in source),default=None),"best_ra":min((s["ra"] for s in source),default=None),
                 "empty_front_rate":int(not source),"initialization_sec":old["measurement"]["timings_sec"].get("initialization"),
                 "init_cost":old.get("init_cost"),"init_ra":old.get("init_ra"),
                 **{k:old[k] for k in ("runtime_sec","candidate_evaluations","scenario_simulations","route_simulations")}}
            rows.append(row)
            eligible=[s for s in r["solutions"] if s["eligible"]]
            front,selected=retained_solutions(eligible)
            searched=[s for s in r["solutions"] if s["search_feasible"]]
            average=lambda values:float(np.mean(values)) if values else None
            row={**identity,"stage":"oos",**quality(ctx["oos"],front),"empty_front_rate":int(not selected),
                 "mean_cost":average([s["cost"]["mean"] for s in selected]),"mean_ra":average([s["ra"]["mean"] for s in selected]),
                 "failure_rate":average([s["failure"]["rate"] for s in searched]),
                 "energy_failure_rate":average([s["energy_failure"]["rate"] for s in searched]),
                 "runtime_sec":r["replay_runtime_sec"],"source_feasible_solutions":len(searched),
                 "candidate_evaluations":r["measurement"]["counts"].get("candidate_evaluations",0),
                 "scenario_simulations":r["measurement"]["counts"].get("scenario_simulations",0)}
            for k in ("num_charge","low_soc_driving_time_min","charging_time_min","charged_energy_kwh"):
                row[k]=average([s[k]["mean"] for s in selected])
            rows.append(row)
            for s in r["solutions"]:
                sr={**identity,"solution_id":s["solution_id"],"search_feasible":s["search_feasible"],"oos_feasible":s["oos_feasible"],
                    "eligible":s["eligible"],"source_cost":s["source_cost"],"source_ra":s["source_ra"],
                    "failure_reasons":s["failure_reason_route_counts"],"routes_by_clusters":s["routes_by_clusters"]}
                for k in ("cost","ra","cost_conditional_on_success","ra_conditional_on_success","failure","energy_failure","num_charge","low_soc_driving_time_min"):
                    sr.update({k+"_"+sub:v for sub,v in s[k].items()})
                solution_rows.append(sr)
    assert len(rows)==1440
    write_csv(directory/"runs.csv",rows); write_csv(directory/"solutions.csv",solution_rows)
    write_json(directory/"reference_contexts.json",contexts); write_json(directory/"input_hashes.json",hashes)
    groups=[]
    for instance in INSTANCES:
        for variant in VARIANTS:
            for stage,metrics in (("search",SEARCH+("initialization_sec","init_cost","init_ra","empty_front_rate")),("oos",OOS+("failure_rate",))):
                rr=[r for r in rows if r["instance"]==instance and r["variant"]==variant and r["stage"]==stage]
                assert len(rr)==30 and len({r["run_id"] for r in rr})==30
                for metric in metrics:
                    seed=stable_seed(20261030,instance,0,variant+stage+metric)
                    values=[r.get(metric) for r in rr]
                    groups.append(dict(instance=instance,variant=variant,stage=stage,metric=metric,
                                       **bootstrap_mean(values,seed)))
    write_csv(directory/"groups.csv",groups)
    tests=paired_statistics(rows); write_csv(directory/"paired_statistics.csv",tests)
    body=format_tables(groups,directory)
    significant=sum(t["significant"] for t in tests)
    notes=("Four configurations, six instances, 30 matched runs per group. Full baseline: 180 verified Table 6 searches; "
           "new variants: 540 searches. Search budget includes preprocessing and initialization. Validation fixes routes and the saved charging policy, "
           "using the same 2,000 independent demands per instance. Bounds pool feasible search and validation objectives across all four configurations. "
           "HV reference is (1.1,1.1); search and validation use separate pooled empirical IGD+ fronts. No common feasible reference is marked missing. "
           "All 2,000 scenarios must succeed for a validation-feasible solution. Empty-front HV/PFS are zero; remaining front statistics are conditional on "
           "valid fronts with sample counts in groups.csv. Front operating means use one deterministic representative per duplicate objective point. "
           "Failure rates include every originally feasible solution, averaged within runs then across runs. Charging counts include depot charging only "
           "when energy added is positive. Low-SOC time integrates driving below 20%, excluding service and charging. Penalized values for failed delivery "
           "are audit objectives, not complete-service economic costs. Confidence intervals use 10,000 run-level BCa resamples, conditional on the fixed "
           "validation set; degenerate intervals are flagged. The 36 paired two-sided Wilcoxon tests compare each variant with full on validation HV/IGD+, "
           "Holm-adjusted together. Positive rank-biserial effects favor full. Differences are rounded only at 12 decimal places for tests; ties/zeros use "
           "100,000 fixed-seed permutations. Fewer than two pairs is marked unavailable and retained in the multiplicity family. "
           "Best marks use unrounded within-table means; operational counts and failure metrics are not marked as quality winners. "
           "Low SOC is a diagnostic, not an added optimization objective. The two tables require booktabs, longtable and amssymb.")
    write_text(directory/"statistical_methods.md",notes+"\n")
    summary=f"## Mechanism ablations: {output.name}\n\n{notes}\n\nHolm-significant paired comparisons: {significant}/36; see effects and directions in paired_statistics.csv.\n\n"+body
    write_text(directory/"analysis.md",summary+"\n")
    write_json(directory/"audit.json",{"search_runs":720,"new_searches":540,"reused":180,"replay_runs":720,
        "saved_solutions":len(solution_rows),"scenario_records":2000*len(solution_rows),"paired_comparisons":36,
        "extra_searches_for_report":0,"extra_simulations_for_report":0})
    if plan.get("analysis_md"):
        target=Path(plan["analysis_md"])
        marker=f"<!-- {output.name}:mechanisms -->"; end=f"<!-- /{output.name}:mechanisms -->"
        import msvcrt
        with target.with_suffix(".md.lock").open("a+b") as lock:
            lock.write(b"0"); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            old=target.read_text(encoding="utf-8") if target.exists() else ""
            block=marker+"\n\n"+summary+"\n"+end
            if marker in old:
                a=old.index(marker); b=old.index(end,a)+len(end); updated=old[:a]+block+old[b:]
            else:
                updated=old+"\n\n"+block+"\n"
            write_text(target,updated)
