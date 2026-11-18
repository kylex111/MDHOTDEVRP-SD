"""Generate out-of-sample HTML/CSV reports and figures without running search."""
from __future__ import annotations

from collections import defaultdict
import html
from pathlib import Path

import numpy as np

from experiments.validation_io import write_text, write_csv, write_json
from .scenario_validation import SCENARIO_COUNTS, GROUP_METRICS, pooled_front

LABELS = {"HV": "HV ↑", "IGD_plus": "IGD+ ↓", "PFS": "PFS ↑", "best_cost": "Best expected cost ↓",
    "best_ra": "Best expected RA ↓", "mean_cost": "Mean front cost", "mean_ra": "Mean front RA",
    "candidate_evaluations": "Candidate evaluations", "scenario_simulations": "Search scenario evaluations",
    "search_runtime_sec": "Search time (s)", "replay_runtime_sec": "Replay time (s)",
    "solution_failure_rate": "Mean OOS failure rate of originally feasible solutions", "source_empty": "Search empty-front rate",
    "oos_empty": "Replay empty-front rate", "front_retention": "Original front retention",
    "HV_change": "OOS HV − in-sample HV", "PFS_change": "PFS change"}
INSTANCE_ORDER = ("c105_21", "r105_21", "rc105_21", "c205_21", "r205_21", "rc205_21")
CSS = """body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:28px;color:#172437;background:#f5f7fa}
main{max-width:1500px;margin:auto}h1{font-size:26px}h2{margin-top:32px;font-size:21px}
.card{background:white;border:1px solid #dbe2eb;border-radius:8px;padding:20px;margin:18px 0}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{border-bottom:1px solid #d9e1ea;padding:9px;text-align:right;white-space:nowrap}
th{background:#eaf0f7;position:sticky;top:0}th:first-child,td:first-child{text-align:left}
.group{background:#f4f7fb;font-weight:bold}.muted,small{color:#586777}a{color:#185ba5}img{max-width:100%}
.bad{color:#a53030}strong{font-weight:650}p,li{line-height:1.7}code{font-family:Consolas,monospace}
"""


def fmt(value, metric=""):
    if value is None:
        return "—"
    if metric in ("solution_failure_rate", "source_empty", "oos_empty", "front_retention"):
        return f"{value*100:.2f}%"
    if metric in ("best_cost", "mean_cost", "candidate_evaluations", "scenario_simulations", "search_runtime_sec", "replay_runtime_sec"):
        return f"{value:,.2f}"
    return f"{value:.4f}"


def cell(summary, metric):
    mean = fmt(summary["mean"], metric)
    sd = fmt(summary["std"], metric)
    return f"{mean} ± {sd}<br><small>n={summary['n']}</small>"


def make_plots(report, runs, groups, prefixes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    palette = dict(zip(SCENARIO_COUNTS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A5ACD")))
    images = []
    for budget in sorted({r["budget_group"] for r in runs}):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        for ax, instance in zip(axes.flat, INSTANCE_ORDER):
            for count in SCENARIO_COUNTS:
                points = [[s["cost"]["mean"], s["ra"]["mean"]] for r in runs
                    if r["budget_group"] == budget and r["instance"] == instance and r["scenario_count"] == count
                    for s in r["solutions"] if s["eligible"]]
                front = pooled_front(points)
                if len(front):
                    ax.plot(front[:, 0], front[:, 1], "o-", markersize=3, linewidth=1,
                            color=palette[count], label=f"S={count}")
            ax.set(title=instance.upper().replace("_21", ""), xlabel="Expected cost", ylabel="Expected RA")
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "No strictly feasible OOS front", transform=ax.transAxes, ha="center")
        fig.suptitle(f"{budget}: pooled strictly feasible OOS fronts")
        name = f"fronts_{budget}.png"
        fig.savefig(report / name, dpi=180)
        fig.savefig(report / name.replace(".png", ".svg"))
        plt.close(fig)
        images.append((name, f"{budget}: pooled OOS nondominated fronts by scenario count"))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        for ax, instance in zip(axes.flat, INSTANCE_ORDER):
            for count in SCENARIO_COUNTS:
                subset = [r for r in prefixes if r["budget_group"] == budget and r["instance"] == instance and r["scenario_count"] == count]
                xs = sorted({r["prefix_n"] for r in subset})
                ys = []
                for n in xs:
                    vals = [r["jaccard_to_full"] for r in subset if r["prefix_n"] == n and r["jaccard_to_full"] is not None]
                    ys.append(float(np.mean(vals)) if vals else np.nan)
                ax.plot(xs, ys, "o-", color=palette[count], label=f"S={count}", markersize=3)
            ax.set(title=instance.upper().replace("_21", ""), xlabel="Independent demand scenarios", ylabel="Mean front ID Jaccard", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle(f"{budget}: prefix fronts compared with the full OOS front (nonempty unions)")
        name = f"stability_{budget}.png"
        fig.savefig(report / name, dpi=180)
        fig.savefig(report / name.replace(".png", ".svg"))
        plt.close(fig)
        images.append((name, f"{budget}: front stability from saved records"))
    return images


def render_report(report, plan, runs, groups, rows, prefixes, final):
    report = Path(report)
    lookup = {(g["budget_group"], g["instance"], g["scenario_count"]): g for g in groups}
    budgets = sorted({g["budget_group"] for g in groups}, reverse=True)
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Scenario reduction: {plan['oos_scenarios']} independent scenarios</title><style>{CSS}</style><main>",
        f"<h1>Scenario reduction: {plan['oos_scenarios']} independent scenarios</h1>",
        f"<p>{"Final report: both budget groups share the same metric reference." if final else "Phase A report: time-budget results only; references will be recomputed after phase C."}</p>",
        f"<p>Reevaluated {len(runs)} searches containing {sum(len(r['solutions']) for r in runs):,} archived solutions. Customer assignments and causal_v1 recourse are fixed.</p>",
        "<div class='card'><p><strong>Definitions</strong>: Main tables include only originally feasible solutions that complete every new scenario. Empty fronts have HV/PFS=0; other metrics are missing, with valid n shown. Tables report mean ± sample standard deviation; 95% intervals appear below.</p>",
        "<p>Failure rates cover originally feasible solutions, including those excluded after replay. Penalized objectives, success-conditional means, and execution before failure are recorded separately. The model does not define the cost of completing service after failure.</p>",
        "<p>1,000 demand scenarios are not 1,000 independent searches. Between-run intervals are conditional on the fixed demand set; zero observed failures do not imply zero failure probability.</p>",
        "<p>R105/R205 and RC105/RC205 have duplicate effective inputs in the current model without time windows. Their requested names are retained.</p></div>",
        "<p><a href='runs.csv'>Run data</a> · <a href='solutions.csv'>Solutions and intervals</a> · <a href='groups.csv'>Group summaries and intervals</a> · <a href='paired_differences.csv'>Paired differences</a> · <a href='front_stability.csv'>Front stability</a> · <a href='quality_contexts.json'>Shared references</a> · <a href='methods.md'>Statistical methods</a></p>"]
    main_metrics = ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "candidate_evaluations")
    for budget in budgets:
        title = "Fixed 1,200 seconds" if budget == "time1200" else "Fixed 2,000 candidate evaluations"
        parts.append(f"<h2>{title}: Table 7 layout</h2><div class='card scroll'><table><thead><tr><th>Scenarios / metric</th>" +
            "".join(f"<th>{x.upper().replace('_21','')}</th>" for x in INSTANCE_ORDER) + "</tr></thead><tbody>")
        for count in SCENARIO_COUNTS:
            parts.append(f"<tr class='group'><td colspan='7'>Search scenarios S={count}</td></tr>")
            for metric in main_metrics:
                parts.append(f"<tr><td>{LABELS[metric]}</td>" + "".join(
                    f"<td>{cell(lookup[(budget, instance, count)]['metrics'][metric], metric)}</td>"
                    for instance in INSTANCE_ORDER) + "</tr>")
        parts.append("</tbody></table></div>")
    parts.append("<h2>Failure rates, front stability, and runtime</h2><div class='card scroll'><table><thead><tr><th>Budget / instance / S</th>" +
        "".join(f"<th>{LABELS[m]}</th>" for m in ("source_empty", "oos_empty", "solution_failure_rate", "front_retention", "mean_cost", "mean_ra", "search_runtime_sec")) + "</tr></thead><tbody>")
    for g in groups:
        parts.append(f"<tr><td>{g['budget_group']} / {g['instance'].upper()} / {g['scenario_count']}</td>" +
            "".join(f"<td>{cell(g['metrics'][m],m)}</td>" for m in
                ("source_empty", "oos_empty", "solution_failure_rate", "front_retention", "mean_cost", "mean_ra", "search_runtime_sec")) + "</tr>")
    parts.append("</tbody></table></div>")
    parts.append("<h2>Wilson 95% intervals for empty-front rates</h2><p>Each condition contains 30 searches. Zero observed failures still have a nonzero upper bound. <a href='empty_front_rates.csv'>Download table</a></p><div class='card scroll'><table><tr><th>Budget / instance / S</th><th>Empty search fronts</th><th>Empty replay fronts</th></tr>")
    for g in groups:
        cells = []
        for name in ("source_empty_rate", "oos_empty_rate"):
            value = g[name]
            cells.append(f"<td>{value['failures']}/{value['n']} ({value['rate']*100:.2f}%)<br><small>[{value['ci95'][0]*100:.2f}%, {value['ci95'][1]*100:.2f}%]</small></td>")
        parts.append(f"<tr><td>{g['budget_group']} / {g['instance'].upper()} / {g['scenario_count']}</td>" + ''.join(cells) + '</tr>')
    parts.append("</table></div>")
    parts.append("<h2>95% confidence intervals</h2><p>BCa intervals use independent runs. Fewer than two valid observations or degenerate samples are marked not estimable. Intervals are marginal, not simultaneous.</p><div class='card scroll'><table><thead><tr><th>Budget / instance / S / metric</th><th>Mean</th><th>95% interval</th><th>Valid n</th><th>Interval status</th></tr></thead><tbody>")
    for g in groups:
        for metric in (*main_metrics[:-1], "solution_failure_rate", "front_retention"):
            v = g["metrics"][metric]
            ci = v["ci95"]
            interval = "—" if ci is None else f"[{fmt(ci[0],metric)}, {fmt(ci[1],metric)}]"
            parts.append(f"<tr><td>{g['budget_group']} / {g['instance'].upper()} / {g['scenario_count']} / {LABELS[metric]}</td><td>{fmt(v['mean'],metric)}</td><td>{interval}</td><td>{v['n']}</td><td>{v['ci_status']}</td></tr>")
    parts.append("</tbody></table></div>")
    rankings = []
    for budget in budgets:
        for instance in INSTANCE_ORDER:
            entries = [lookup[(budget, instance, s)] for s in SCENARIO_COUNTS]
            for metric in ("source_HV", "HV", "IGD_plus"):
                valid = [g for g in entries if g['metrics'][metric]['mean'] is not None]
                valid.sort(key=lambda g: g['metrics'][metric]['mean'], reverse=metric != 'IGD_plus')
                rankings.append({"budget_group": budget, "instance": instance, "metric": metric,
                                 "ordered_S": [g['scenario_count'] for g in valid]})
    write_csv(report / "rankings.csv", rankings)
    parts.append("<h2>Scenario-group ranking changes</h2><p>Ranks describe group means; tied values are not interpreted as differences.</p><div class='card scroll'><table><tr><th>Budget / instance</th><th>In-sample HV</th><th>OOS HV</th><th>OOS IGD+</th></tr>")
    for budget in budgets:
        for instance in INSTANCE_ORDER:
            entries = {r['metric']:r['ordered_S'] for r in rankings if r['budget_group']==budget and r['instance']==instance}
            parts.append(f"<tr><td>{budget} / {instance.upper()}</td>" + ''.join(f"<td>{' → '.join(map(str,entries[m])) or '—'}</td>" for m in ('source_HV','HV','IGD_plus')) + '</tr>')
    parts.append("</table></div>")
    for name, label in make_plots(report, runs, groups, prefixes):
        parts.append(f"<div class='card'><h2>{label}</h2><img src='{name}' alt='{label}'></div>")
    parts.append("</main></html>")
    write_text(report / "comparison.html", "\n".join(parts))
    methods = """# Out-of-sample evaluation and statistical methods

Reevaluate fixed plans on shared, independent, equally weighted demand vectors for each instance. Reset vehicle state per scenario; do not select initialization directions or run neighborhood search.

Main tables require search_feasible=true and completed=true in every OOS scenario. Replay must not replace an originally empty search front. Dominance tolerance is 1e-9. Preserve independent run IDs.

Record both penalized objective means and success-conditional means. The model does not define the cost of completing service after failure; execution before failure is diagnostic only. Single-solution mean intervals use approximate Student-t 95% intervals, with the actual success count for conditional means. Failure rates use Wilson 95% intervals over all independent demand vectors.

Each condition has 30 independent searches. Main tables show means and sample standard deviations. Mean 95% intervals use 10,000 BCa resamples at the run level. Do not estimate intervals with fewer than two valid values or degenerate resamples. Paired differences use common run IDs with both values defined. Between-run intervals are conditional on the fixed OOS demand set; single-solution scenario intervals separately reflect demand sampling error. Intervals are marginal, with no multiple-significance claim.

Empty fronts have HV/PFS=0. IGD+, best cost/RA, and front means are undefined; conditional summaries show valid n. Best cost and best RA within a run may come from different solutions. Average failure rates equally across originally feasible solutions within each run, then equally across runs. Runs with no originally feasible solution have a missing solution failure rate and a separately reported empty-front rate.

Use one normalization range per instance across feasible in-sample, full OOS, and prefix-diagnostic objectives. The HV reference point is (1.1, 1.1). The empirical IGD+ reference set pools strictly feasible full-OOS objectives from all groups in the report. IGD+ is missing if no common feasible reference exists. Phase A uses provisional references; the final report recomputes them across both budget groups.

Front retention uses original nondominated solution IDs. Distinct IDs with identical objectives retain their identity; PFS counts unique objective points. Prefix analysis reuses the first 100/250/500/1,000 saved scenarios without new simulation. Compute Jaccard similarity against the full OOS front using solution IDs; it is undefined for an empty union. Every prefix uses zero observed failures as its feasibility criterion.

1,000 scenarios cannot establish zero true failure probability or rule out rare events. Do not assume 10 scenarios are best. R105/R205 and RC105/RC205 have duplicate effective inputs in the current model without time windows.

Per-scenario records, routes, demands, configurations, and hashes remain in the batch directory. Reports can be rebuilt from saved data without repeating search or simulation.
"""
    write_text(report / "methods.md", methods)
    # Paper tables and the web report share the same statistics.
    md = ["# Out-of-sample scenario reduction results", "", "Mean ± sample standard deviation; see CSV for valid n.", ""]
    tex = []
    for budget in budgets:
        md.extend([f"## {budget}", "", "| S / metric | " + " | ".join(INSTANCE_ORDER) + " |",
                   "|---|" + "---:|"*6])
        tex.append("\\begin{tabular}{llrrrrrr}\n\\hline\nS & Metric & C105 & R105 & RC105 & C205 & R205 & RC205 \\\\\n\\hline")
        for count in SCENARIO_COUNTS:
            for metric in main_metrics:
                vals = [lookup[(budget, i, count)]["metrics"][metric] for i in INSTANCE_ORDER]
                cells = [f"{fmt(v['mean'],metric)} ± {fmt(v['std'],metric)}" for v in vals]
                md.append(f"| {count} / {LABELS[metric]} | " + " | ".join(cells) + " |")
                tex.append(f"{count} & {metric.replace('_', ' ')} & " + " & ".join(c.replace('±', '$\\pm$').replace('—','--') for c in cells) + " \\\\")
        tex.append("\\hline\n\\end{tabular}\n")
    write_text(report / "table7.md", "\n".join(md))
    write_text(report / "table7.tex", "\n".join(tex))
