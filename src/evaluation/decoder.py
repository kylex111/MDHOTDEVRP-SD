"""Charging preparation and deterministic decoding; see causal for stochastic recourse."""

from __future__ import annotations


class Decoder:
    def __init__(
        self,
        problem,
        energy_model,
        ra_model,
        ra_safe,
        ra_risk,
        margin_ratio=0.1,
        radius_km=3.0,
    ):
        self.problem = problem
        self.charging_nodes = list(self.problem.css)
        self.energy_model = energy_model
        self.margin_ratio = margin_ratio
        self.radius_km = radius_km
        self.ra_model = ra_model
        self.ra_safe = ra_safe
        self.ra_risk = ra_risk
        self.EPS = 1e-09

    def _build_anchor_cs_map(self, route):
        """Map each node to the nearest or lowest-energy station within 3 km."""
        anchor_cs = {}
        for n in route:
            best = None
            best_d = float("inf")
            for cs in self.problem.css:
                d = self.problem.distance_matrix[n, cs]
                if d <= self.radius_km and d < best_d:
                    best = cs
                    best_d = d
            if best is not None:
                anchor_cs[n] = best
        return anchor_cs

    def _estimate_partial_charge(
        self, cs_index_in_route, route, Q_remain, departure_time, load
    ):
        """Estimate energy from curr_cs through the remaining route to the final depot.

        Charge fully if required energy exceeds Q_max; otherwise charge as needed.
        Return required_charge and charge_time."""
        Q_max = self.problem.soc_max
        margin = self.margin_ratio * Q_max
        total_energy_needed = 0.0
        temp_departure = departure_time
        temp_load = load
        for j in range(cs_index_in_route, len(route) - 1):
            prev_node = route[j]
            curr_node = route[j + 1]
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, temp_departure, temp_load
            )
            total_energy_needed += energy
            temp_departure += travel_time
            service_time = self.problem.service_time[curr_node]
            demand = self.problem.demand[curr_node]
            temp_departure += service_time
            temp_load -= demand
        total_energy_needed += margin
        theoretical_required = max(0.0, total_energy_needed - Q_remain)
        required_charge = min(Q_max - Q_remain, theoretical_required)
        charge_time = self.energy_model.calculate_charging_time(required_charge)
        return (required_charge, charge_time)

    def _can_reach_final_depot(
        self, cs_index_in_route, route, Q_after_charge, departure_time, load
    ):
        """Check whether the remaining route can reach the final depot without another charging stop."""
        "Check whether direct-to-depot mode can safely skip subsequent current-node-to-station checks."
        Q = Q_after_charge
        t = departure_time
        l = load
        for i in range(cs_index_in_route, len(route) - 1):
            u, v = (route[i], route[i + 1])
            dist = self.problem.distance_matrix[u, v]
            e, dt, _ = self.energy_model.calculate_one_node_energy_time(dist, t, l)
            if Q + self.EPS < e:
                return False
            Q -= e
            t += dt
            t += self.problem.service_time[v]
            l -= self.problem.demand[v]
        return True

    def _to_nearest_cs(self, prev_node, Q_at_prev, t_depart_prev, load_prev):
        dm = self.problem.distance_matrix
        nearest_node = min(self.charging_nodes, key=lambda n: dm[prev_node, n])
        dist = dm[prev_node, nearest_node]
        e, dt, _ = self.energy_model.calculate_one_node_energy_time(
            dist, t_depart_prev, load_prev
        )
        if Q_at_prev + self.EPS >= e:
            return (nearest_node, dist, e, dt)
        return None

    def _ra_mode(self, ra):
        """Classify the current RA as safe, opportunity, or risk."""
        if self.ra_model is None:
            return "safe_mode"
        if ra >= self.ra_risk:
            return "risk_mode"
        elif ra < self.ra_safe:
            return "safe_mode"
        else:
            return "opportunity_mode"

    def decode_deterministic(self, route, load, departure_time=0):
        """Decode with lookahead, backtracking station insertion, and low_soc_mode diagnostics.

        If the next target remains unreachable after repair, return a large penalty
        instead of retrying indefinitely or raising an error."""
        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False
        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])
        t_since_charge = 0.0
        Q_ref = Q
        total_ra = 0.0
        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0
        state_log = []
        i = 1
        BIG_PENALTY = 1000000.0
        WATCHDOG_LIMIT = 200
        watchdog = 0
        tried_pairs = set()
        skip_charge_once = False
        skip_charge_cs_id = None

        def penalty_exit():
            return (
                route,
                BIG_PENALTY,
                BIG_PENALTY,
                BIG_PENALTY,
                BIG_PENALTY,
                999,
                0.0,
                departure_time,
                load,
                BIG_PENALTY,
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return penalty_exit()
            prev_node = route[i - 1]
            curr_node = route[i]
            state_log.append(
                {
                    "prev_i": i - 1,
                    "route": route[:],
                    "Q": Q,
                    "Q_ref": Q_ref,
                    "t_since_charge": t_since_charge,
                    "departure_time": departure_time,
                    "load": load,
                    "total_travel_time": total_travel_time,
                    "total_service_time": total_service_time,
                    "total_charging_time": total_charging_time,
                    "total_dist": total_dist,
                    "total_ra": total_ra,
                    "lock_to_depot": lock_to_depot,
                }
            )
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_ref - Q)
                Ttr = t_since_charge
                hour_of_day = departure_time / 60.0 % 24.0
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                ra = 0.0
                mode = "safe_mode"
            if not lock_to_depot and mode == "risk_mode":
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue
            if (
                not lock_to_depot
                and mode == "opportunity_mode"
                and (curr_node in self.cs_in_3km_node)
            ):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = route[i - 1] == cs_id if i - 1 >= 0 else False
                already_after = i + 1 < len(route) and route[i + 1] == cs_id
                inserted_now = False
                if not already_before and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None
                    d_prev_cs = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr = self.problem.distance_matrix[cs_id, curr_node]
                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs = self.problem.distance_matrix[curr_node, cs_id]
                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next = self.problem.distance_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs
                    if insert_before_cost + self.EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                    else:
                        route.insert(i + 1, cs_id)
                        inserted_now = True
                if inserted_now:
                    continue
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = (
                self.energy_model.calculate_one_node_energy_time(
                    dist, departure_time, load
                )
            )
            if Q + EPS < energy_needed:
                inserted = False
                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]
                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra_total = snap["total_ra"]
                    _snap_lock = snap["lock_to_depot"]
                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(
                        prev_node2, snap_Q, snap_t, snap_load
                    )
                    if cs_result is None:
                        continue
                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result
                    if cs_id == prev_node2:
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        continue
                    if (prev_i, cs_id) in tried_pairs:
                        return penalty_exit()
                    tried_pairs.add((prev_i, cs_id))
                    snap_route.insert(prev_i + 1, cs_id)
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy
                    required_charge, charge_time = self._estimate_partial_charge(
                        prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load
                    )
                    Q = min(self.problem.soc_max, Q_arrive_cs + required_charge)
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load
                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_charging_time = snap_ct + charge_time
                    total_dist = snap_dist + to_cs_dist
                    num_charge += 1
                    t_since_charge = 0.0
                    Q_ref = Q
                    if getattr(self, "ra_model", None) is not None:
                        Ttr0 = snap_t_since
                        hour0 = snap_t / 60.0 % 24.0
                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy
                        seg_ra = self.ra_model.SUMR(
                            Qtr_after, Ttr0, hour0
                        ) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        total_ra = snap_ra_total + seg_ra
                    else:
                        total_ra = snap_ra_total
                    if self._can_reach_final_depot(
                        prev_i + 1, snap_route, Q, departure_time, load
                    ):
                        lock_to_depot = True
                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[: prev_i + 1]
                    inserted = True
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                        energy2, _, _ = (
                            self.energy_model.calculate_one_node_energy_time(
                                dist2, departure_time, load
                            )
                        )
                        if Q + EPS < energy2:
                            return penalty_exit()
                    break
                if not inserted:
                    return penalty_exit()
                continue
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = departure_time / 60.0 % 24.0
                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = self.ra_model.SUMR(
                    Qtr_after, Ttr0, hour0
                ) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                total_ra += seg_ra
            arrival_time = departure_time + travel_time
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            Q -= energy_needed
            load -= self.problem.demand[curr_node]
            total_travel_time += travel_time
            total_service_time += service_time
            total_dist += dist
            t_since_charge += travel_time
            if curr_node in self.problem.css:
                if skip_charge_once and curr_node == skip_charge_cs_id:
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge, charge_time = self._estimate_partial_charge(
                        i, route, Q, departure_time, load
                    )
                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time
                        num_charge += 1
                    t_since_charge = 0.0
                    Q_ref = Q
                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True
            i += 1
        return (
            route,
            total_dist,
            total_travel_time,
            total_charging_time,
            total_service_time,
            num_charge,
            Q,
            departure_time,
            load,
            total_ra,
        )
