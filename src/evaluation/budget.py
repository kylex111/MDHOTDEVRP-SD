"""Per-solve evaluation budgets, phase timing, and simulation counts."""

from __future__ import annotations
from time import perf_counter


class BudgetExceeded(RuntimeError):
    def __init__(self, reason, measurement=None):
        self.reason = reason
        self.measurement = measurement
        super().__init__(reason)


class Measurement:
    def __init__(self, max_eval=None, max_time_sec=None):
        self.started = perf_counter()
        self.max_eval, self.max_time_sec = (max_eval, max_time_sec)
        self.phase = "preprocessing"
        self.phase_started = self.started
        self.timings = {}
        self.counts = {}

    def set_phase(self, phase):
        now = perf_counter()
        self.timings[self.phase] = (
            self.timings.get(self.phase, 0.0) + now - self.phase_started
        )
        self.phase, self.phase_started = (phase, now)

    def increment(self, field):
        self.counts[field] = self.counts.get(field, 0) + 1
        name = f"{self.phase}_{field}"
        self.counts[name] = self.counts.get(name, 0) + 1

    def before_candidate(self):
        if (
            self.max_eval is not None
            and self.counts.get("candidate_evaluations", 0) >= self.max_eval
        ):
            raise BudgetExceeded("max_eval", self.snapshot())
        if (
            self.max_time_sec is not None
            and perf_counter() - self.started >= self.max_time_sec
        ):
            raise BudgetExceeded("max_time", self.snapshot())
        self.increment("candidate_evaluations")

    def snapshot(self):
        now = perf_counter()
        timing = dict(self.timings)
        timing[self.phase] = timing.get(self.phase, 0.0) + now - self.phase_started
        return {
            "elapsed_sec": now - self.started,
            "timings_sec": timing,
            "counts": dict(self.counts),
            "budget_includes_initialization": True,
            "candidate_definition": "one scenario-aggregated evaluation request, including initialization and repeated routes",
            "scenario_definition": "one actual solution/scenario simulation started",
            "route_definition": "one nonempty route decode started",
            "time_boundary": "preprocessing + initialization + search; an in-flight candidate is atomic",
        }
