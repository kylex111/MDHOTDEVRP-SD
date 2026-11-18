"""Reveal stochastic demand on arrival and apply charging and replenishment recourse."""

from __future__ import annotations
import math
from evaluation.charging import low_soc_duration

CHARGE_BUCKETS = ("opportunistic", "risk", "repair", "natural")


def decode_causal(
    decoder,
    route,
    load,
    demand=None,
    departure_time=0.0,
    *,
    capture=False,
    capture_motion=False,
):
    p, energy, ra_model = (decoder.problem, decoder.energy_model, decoder.ra_model)
    charging_policy = getattr(decoder, "charging_policy", "ra_based")
    soc_threshold = getattr(decoder, "soc_threshold", 0.20)
    collect_low_soc = getattr(decoder, "collect_low_soc", False)
    demand = p.demand if demand is None else demand
    pending = list(route[1:])
    if not route:
        raise ValueError("An executed route needs a starting depot.")
    prev = route[0]
    executed = [prev]
    customers, depots, stations = (set(p.customers), set(p.depots), set(p.css))
    # Depots support charging but retain their depot type for separate replenishment handling.
    charging_nodes = depots | stations
    capacity, battery, eps = (float(p.vehicle_capacity), float(p.soc_max), decoder.EPS)
    q, q_ref, since_charge = (battery, battery, 0.0)
    travel = charging = service = distance = total_ra = 0.0
    restocks = 0
    known, served, trace = ({}, set(), [])
    stats = {
        f"{prefix}_{bucket}": 0.0
        for prefix in ("num_charge", "charging_time", "charged_energy")
        for bucket in ("total", *CHARGE_BUCKETS)
    }
    diagnostics = {
        "completed": False,
        "failure_reason": None,
        "energy_failure": False,
        "ief_events": 0,
        "sef_events": 0,
        "load_failure_events": 0,
        "min_soc_fraction": 1.0,
        "historical_backtracks": 0,
        "policy": "causal_v1",
        "trace": trace,
    }
    if collect_low_soc:
        diagnostics["low_soc_driving_time_min"] = 0.0
        diagnostics["low_soc_threshold"] = 0.20
    if capture_motion:
        diagnostics["motion_segments"] = []
    charge_tags = ["natural"] * len(pending)
    restock_for = None

    def event(kind, **extra):
        if capture:
            trace.append(
                {
                    "event": kind,
                    "node": int(prev),
                    "time_min": float(departure_time),
                    "battery_kwh": float(q),
                    "load_kg": float(load),
                    **extra,
                }
            )

    def finish(reason=None, energy_failure=False):
        diagnostics.update(
            completed=reason is None,
            failure_reason=reason,
            energy_failure=energy_failure,
            served_customers=sorted(served),
            num_restock=restocks,
            actual_travel_time_min=travel,
            actual_charging_time_min=charging,
            actual_service_time_min=service,
            actual_distance_km=distance,
            actual_executed_ra=total_ra,
        )
        decoder.last_decode_diagnostics = diagnostics
        decoder.last_decode_stats = stats
        event("complete" if reason is None else "failure", reason=reason)
        if reason is not None:
            return (
                executed,
                1000000.0,
                1000000.0,
                1000000.0,
                1000000.0,
                int(stats["num_charge_total"]),
                q,
                departure_time,
                load,
                1000000.0,
                restocks,
                stats,
            )
        return (
            executed,
            distance,
            travel,
            charging,
            service,
            int(stats["num_charge_total"]),
            q,
            departure_time,
            load,
            total_ra,
            restocks,
            stats,
        )

    def segment(a, b, t, payload, details=None):
        d = float(p.distance_matrix[a, b])
        kwargs = {"segments": details} if details is not None else {}
        e, dt, _ = energy.calculate_one_node_energy_time(d, t, payload, **kwargs)
        return (d, float(e), float(dt))

    def nearest_charging_node(node):
        return min(
            (dest for dest in charging_nodes if dest != node),
            key=lambda dest: (float(p.distance_matrix[node, dest]), dest),
            default=None,
        )

    def reachable_cs():
        cs = nearest_charging_node(prev)
        if cs is None:
            return None
        d, e, dt = segment(prev, cs, departure_time, load)
        return (d, cs, e, dt) if e <= q + eps else None

    def needs_sef(nxt, required, dt):
        if nxt in charging_nodes:
            return False
        arrival = departure_time + dt
        if nxt in known:
            actual = known[nxt]
            # Use revealed demand for the actual branch; leave immediately to replenish if service is not possible.
            departures = (
                [(arrival, load)]
                if load + eps < actual
                else [(arrival + p.service_time[nxt], max(0.0, load - actual))]
            )
        else:
            # Do not read demand[nxt] before arrival. Both possible departure times must be safe.
            # Use current load conservatively for the service branch, without using unrevealed delivery quantities.
            departures = [(arrival, load), (arrival + p.service_time[nxt], load)]
        dest = nearest_charging_node(nxt)
        if dest is None:
            return True
        for when, payload in departures:
            escape = segment(nxt, dest, when, payload)[1]
            if q - required + eps < escape:
                return True
        return False

    def charge(bucket):
        nonlocal q, q_ref, since_charge, charging, departure_time
        amount = max(0.0, battery - q)
        duration = float(energy.calculate_charging_time(amount))
        if amount > eps:
            for category in ("total", bucket):
                stats[f"num_charge_{category}"] += 1
                stats[f"charging_time_{category}"] += duration
                stats[f"charged_energy_{category}"] += amount
            charging += duration
            departure_time += duration
            q = battery
            event("charge", bucket=bucket, charged_kwh=amount, duration_min=duration)
        q_ref, since_charge = (q, 0.0)

    if not 0 <= load <= capacity + eps:
        return finish("INVALID_INITIAL_LOAD")
    event("depart")
    max_visits = max(1000, 20 * len(route))
    for _ in range(max_visits):
        if not pending:
            required_customers = set(route) & customers
            if served != required_customers:
                return finish("UNSERVED_CUSTOMERS")
            return finish()
        nxt = pending[0]
        details = [] if capture_motion or collect_low_soc else None
        d, required, dt = segment(prev, nxt, departure_time, load, details)
        ief = required > q + eps
        sef = not ief and needs_sef(nxt, required, dt)
        hour = (departure_time / 60.0 + energy.start_hour) % 24.0
        instantaneous = ra_model.R_instant(max(0.0, q_ref - q), since_charge, hour)
        mode = decoder._ra_mode(instantaneous)
        option = reachable_cs()
        at_full_cs = prev in charging_nodes and q >= battery - eps
        bucket = None
        if ief:
            diagnostics["ief_events"] += 1
            bucket = "repair"
        elif sef:
            diagnostics["sef_events"] += 1
            bucket = "repair"
        elif (charging_policy == "soc_threshold" and nxt not in charging_nodes
              and (not at_full_cs) and q / battery < soc_threshold):
            bucket = "risk"
        elif charging_policy == "ra_based" and nxt not in charging_nodes and (not at_full_cs) and (mode == "risk_mode"):
            bucket = "risk"
        elif (
            nxt not in charging_nodes
            and (not at_full_cs)
            and charging_policy == "ra_based"
            and (mode == "opportunity_mode")
            and (option is not None)
            and (p.distance_matrix[option[1], nxt] <= decoder.radius_km)
        ):
            bucket = "opportunistic"
        if bucket is not None:
            if prev in depots and q < battery - eps:
                # Replenishment and depot service are complete; load is the updated payload.
                # Charge in place without adding another depot visit; recheck energy after the charging delay.
                event("depot_charge_decision", next_node=int(nxt), trigger=bucket,
                      ief=ief, sef=sef)
                charge(bucket)
                continue
            if option is None:
                if ief:
                    return finish("NO_REACHABLE_CS", True)
                if sef:
                    return finish("NO_SAFE_CONTINUATION", True)
            elif option[1] != nxt:
                event(
                    "charge_decision",
                    next_node=int(nxt),
                    station=int(option[1]),
                    trigger=bucket,
                    ief=ief,
                    sef=sef,
                )
                pending.insert(0, option[1])
                charge_tags.insert(0, bucket)
                continue
        if ief:
            return finish("UNREACHABLE_NEXT_NODE", True)
        before = max(0.0, q_ref - q)
        total_ra += ra_model.SUMR(
            before + required, since_charge, hour
        ) - ra_model.SUMR(before, since_charge, hour)
        if details is not None:
            segment_q = q
            for part in details:
                next_q = segment_q - part["energy_kwh"]
                if collect_low_soc:
                    diagnostics["low_soc_driving_time_min"] += low_soc_duration(segment_q, next_q, battery * 0.20, part["duration_min"])
                if capture_motion:
                    diagnostics["motion_segments"].append(
                    {
                        "from_node": int(prev),
                        "to_node": int(nxt),
                        "load_kg": float(load),
                        "battery_capacity_kwh": battery,
                        "battery_start_kwh": segment_q,
                        "battery_end_kwh": next_q,
                        "soc_start": segment_q / battery,
                        "soc_end": next_q / battery,
                        **part,
                    }
                )
                segment_q = next_q
        q -= required
        diagnostics["min_soc_fraction"] = min(
            diagnostics["min_soc_fraction"], q / battery
        )
        departure_time += dt
        since_charge += dt
        travel += dt
        distance += d
        prev = pending.pop(0)
        tag = charge_tags.pop(0)
        executed.append(prev)
        event("arrive")
        if q < -eps:
            return finish("NEGATIVE_Q", True)
        if prev in customers:
            if prev not in known:
                known[prev] = float(demand[prev])
                event("reveal", demand_kg=known[prev])
            actual = known[prev]
            if not math.isfinite(actual) or actual < 0:
                return finish("INVALID_CUSTOMER_DEMAND")
            if actual > capacity + eps:
                return finish("DEMAND_EXCEEDS_CAPACITY")
            if load + eps < actual:
                diagnostics["load_failure_events"] += 1
                depot = min(depots, key=lambda x: p.distance_matrix[prev, x])
                pending[:0] = [depot, prev]
                charge_tags[:0] = ["natural", "natural"]
                restock_for = prev
                event("restock_decision", depot=int(depot), revealed_demand_kg=actual)
                continue
            if prev in served:
                return finish("DUPLICATE_CUSTOMER_SERVICE")
            load = max(0.0, load - actual)
            served.add(prev)
            event("service", demand_kg=actual)
        elif prev in depots and restock_for is not None:
            tail = set(pending) & customers - served
            target = sum((known.get(node, float(p.demand[node])) for node in tail))
            load = min(capacity, max(0.0, target))
            restocks += 1
            restock_for = None
            event("restock", target_load_kg=load)
        node_service = float(p.service_time[prev])
        departure_time += node_service
        service += node_service
        if prev in stations or (prev in depots and tag != "natural"):
            # Charge at a depot inserted specifically for charging, as at a station; otherwise charge as needed for departure.
            charge(tag)
        if prev in customers and load <= eps and pending and (pending[0] in customers):
            depot = min(depots, key=lambda x: p.distance_matrix[prev, x])
            restock_for = pending[0]
            pending.insert(0, depot)
            charge_tags.insert(0, "natural")
            event("empty_restock_decision", depot=int(depot))
    return finish("WATCHDOG")
