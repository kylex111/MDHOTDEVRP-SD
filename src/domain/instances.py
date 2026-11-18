"""Read instances, build problems, and map customer IDs to global node indices."""

from __future__ import annotations
from typing import List
from domain.nodes import Node, NodeType


def read_instance(path: str) -> List[Node]:
    """Read an instance into Node objects with indices 0..n-1.

    Node rows contain at least eight columns:
      StringID Type(d/c/f) x y demand ready due service
    Skip blank lines, StringID headers, and comments starting with #."""
    nodes: List[Node] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("StringID"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            node_type_str = parts[1].lower()
            if node_type_str not in ("d", "c", "f"):
                continue
            if len(parts) < 8:
                raise ValueError(
                    f"Bad node line at {path}:{ln} (expect >= 8 cols): {raw.rstrip()}"
                )
            try:
                node = Node(
                    idx=len(nodes),
                    name=parts[0],
                    type=NodeType.from_str(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    demand=float(parts[4]),
                    ready_time=float(parts[5]),
                    due_time=float(parts[6]),
                    service_time=float(parts[7]),
                )
            except Exception as e:
                raise ValueError(f"Parse error at {path}:{ln}: {raw.rstrip()}") from e
            nodes.append(node)
    if not nodes:
        raise ValueError(f"No nodes parsed from {path!r}")
    return nodes


from typing import Dict, Tuple
from domain.problem import EVRPProblem, build_evrp_problem


def build_customer_mappings(
    customers_global_indices: list[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    customer_id: 1..n_customers
    cust_id_to_global: customer_id -> global_idx
    global_to_cust_id: global_idx -> customer_id
    """
    cust_id_to_global = {
        i + 1: customers_global_indices[i] for i in range(len(customers_global_indices))
    }
    global_to_cust_id = {
        customers_global_indices[i]: i + 1 for i in range(len(customers_global_indices))
    }
    return (cust_id_to_global, global_to_cust_id)


def load_problem(
    path: str, ev_params: Dict, *, return_mappings: bool = False
) -> EVRPProblem | Tuple[EVRPProblem, Dict]:
    """Build an EVRPProblem from an instance path and ev_params, optionally returning the customer-to-global index mapping."""
    nodes = read_instance(path)
    problem = build_evrp_problem(nodes, ev_params)
    if not return_mappings:
        return problem
    cust_id_to_global, global_to_cust_id = build_customer_mappings(problem.customers)
    mappings = {
        "cust_id_to_global": cust_id_to_global,
        "global_to_cust_id": global_to_cust_id,
    }
    return (problem, mappings)
