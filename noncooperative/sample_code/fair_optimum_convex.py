import sys
sys.path.append('../')
import os
from qroute.routing.flow_allocator import FlowAllocator
from qroute.routing.allocator import _make_edge_key
from qroute.routing.entanglement_manipulation import werner_param_from_fidelity
from graph_tool.all import Graph

import itertools
import multiprocessing
import pickle
from datetime import datetime

MIN_FLOW = 1e-8

NODE_PAIR_PATH_COUNTS = {
    (0, 1): 15,
    (0, 2): 13,
    (0, 3): 14,
    (0, 4): 15,
    (0, 5): 15,
    (0, 6): 15,
    (0, 7): 14,
    (1, 2): 15,
    (1, 3): 13,
    (1, 4): 13,
    (1, 5): 17,
    (1, 6): 18,
    (1, 7): 16,
    (2, 3): 14,
    (2, 4): 15,
    (2, 5): 15,
    (2, 6): 15,
    (2, 7): 14,
    (3, 4): 13,
    (3, 5): 16,
    (3, 6): 16,
    (3, 7): 17,
    (4, 5): 18,
    (4, 6): 17,
    (4, 7): 16,
    (5, 6): 13,
    (5, 7): 13,
    (6, 7): 13,
}


def evaluate_subset(subset_indices, node_pair, total_paths):
    """
    subset_indices: indices of paths to keep
    returns: (avg_fid, included_path_tuple)
    """
    CAP = 1000
    P_BELL = 1.0
    P_WERNER = werner_param_from_fidelity(0.95)

    g = Graph(
        {0: [1, 2, 5], 1: [3, 4], 2: [4, 6], 3: [4, 7], 5: [6, 7], 6: [7]},
        directed=False,
    )
    bell_state_edges = [(0, 5), (1, 4), (2, 4), (2, 6), (5, 6), (5, 7)]

    edge_werner_param = {}
    edge_capacity = {}
    for edge in g.edges():
        edge_key = _make_edge_key(edge)
        edge_werner_param[edge_key] = (
            P_BELL if edge_key in bell_state_edges else P_WERNER
        )
        edge_capacity[edge_key] = 1.0  # TODO CAP

    allocator = FlowAllocator(
        g, edge_werner_param, edge_capacity, [(node_pair[0], node_pair[1], 3)]
    )

    all_indices = set(range(total_paths))
    excluded = list(all_indices - set(subset_indices))

    try:
        f_nash = allocator.optimize_flow_NEW(
            nash_flow=True, print_path_fidelities=False, excluded_path_indices=excluded
        )
        return (f_nash, tuple(sorted(subset_indices)))
    except Exception:
        return (0, tuple(sorted(subset_indices)))  # treat failures as low score


def wrapped_eval(args):
    excluded_paths, node_pair, total_paths = args
    all_indices = set(range(total_paths))
    included = list(all_indices - set(excluded_paths))
    return evaluate_subset(included, node_pair, total_paths)


def run_exclusion_search(
    exclusion_range, save_path=None, node_pair=(3, 6), total_paths=16
):
    print("▶ Building job list...")
    all_jobs = [
        (excluded_paths, node_pair, total_paths)
        for k in exclusion_range
        for excluded_paths in itertools.combinations(range(total_paths), k)
    ]

    print(
        f"▶ Running search over {len(all_jobs)} subsets (exclusions: {exclusion_range})"
    )
    start = datetime.now()
    with multiprocessing.Pool() as pool:
        results = pool.map(wrapped_eval, all_jobs)
    print("▶ Done. Duration:", datetime.now() - start)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"▶ Results saved to: {save_path}")

    best = max(results, key=lambda x: x[0])
    print(f"▶ Best fidelity: {best[0]:.6f} with included paths: {best[1]}")


# if __name__ == "__main__":
#     print(datetime.now())
#     for node_pair, total_paths in NODE_PAIR_PATH_COUNTS.items():
#         for exclusion_size in range(1, total_paths):  # valid exclusions
#             save_file = (
#                 f"main_paper/figure_2/fair_optimum_data/"
#                 f"fair_optimum_exclusion_pair_{node_pair[0]}_{node_pair[1]}_exclusions_{exclusion_size}.pkl"
#             )
#             if not os.path.exists(save_file):
#                 print(
#                     f"▶ Node pair: {node_pair}, Exclusions: {exclusion_size}/{total_paths}"
#                 )
#                 run_exclusion_search(
#                     [exclusion_size],
#                     save_path=save_file,
#                     node_pair=node_pair,
#                     total_paths=total_paths,
#                 )
#     print(datetime.now())
if __name__ == "__main__":
    print(datetime.now())

    # node_pair = list(NODE_PAIR_PATH_COUNTS.keys())[i] # UP TO 9 good
    node_pair = (4, 7)
    total_paths = NODE_PAIR_PATH_COUNTS[node_pair]
    for exclusion_size in range(1, total_paths):  # valid exclusions
        save_file = (
            f"main_paper/figure_2/fair_optimum_data/"
            f"fair_optimum_exclusion_pair_{node_pair[0]}_{node_pair[1]}_exclusions_{exclusion_size}.pkl"
        )
        if os.path.exists(save_file):
            print(f"⏩ Skipping existing file for exclusions = {exclusion_size}")
            continue

        print(f"▶ Node pair: {node_pair}, Exclusions: {exclusion_size}/{total_paths}")
        run_exclusion_search(
            [exclusion_size],
            save_path=save_file,
            node_pair=node_pair,
            total_paths=total_paths,
        )
    print(datetime.now())
