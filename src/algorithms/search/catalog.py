"""Active shaking and VND neighborhoods, listed in execution order."""

GB_SHAKING = [
    "relocate_boundary",
    "merge_gb_intra_route",
    "relocate_gb_inter_route",
    "relocate_gb_intra_route",
]
GB_VND = [
    "relocate_node_intra_gb",
    "reverse_node_intra_gb",
    "relocate_node_inter_gb",
    "swap_node_inter_gb",
    "relocate_node_inter_route",
    "relocate_boundary",
]
NODE_SHAKING = [
    "relocate_boundary",
    "relocate_node_inter_route",
    "relocate_node_intra_route",
]
NODE_VND = [
    "relocate_node_intra_route",
    "reverse_node_intra_route",
    "swap_node_intra_route",
    "relocate_node_inter_route",
    "relocate_boundary",
]
