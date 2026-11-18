"""Cost parameters and components; charge travel, service, and charging time using the existing formulas."""

from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass


@dataclass
class CostParams:
    c1: float = 120.0
    c2: float = 0.5
    c3: float = 0.3
    c4: float = 0.6


def calculate_cost(ev_number, travel_time, charging_time, service_time, params):
    CostDetail = namedtuple(
        "CostDetail", ["dispatch_cost", "travel_cost", "service_cost", "charging_cost"]
    )
    dispatch_cost = params.c1 * ev_number
    travel_cost = params.c2 * travel_time
    service_cost = params.c3 * service_time
    charging_cost = params.c4 * charging_time
    cost_detail = CostDetail(dispatch_cost, travel_cost, service_cost, charging_cost)
    total_cost = sum(cost_detail)
    return (total_cost, cost_detail)
