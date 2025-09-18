from tabulate import tabulate
from typing import TYPE_CHECKING, List

from qroute.routing.entanglement_manipulation import fidelity_from_werner_param


def print_routing_plan_statistics(
    allocator, show_empty_paths=True, simulate_nonempty_path=True
):
    """
    Prints a formatted table of the current routing plan statistics for each request.

    Includes the path, load, and effective Fidelity (f_eff) for each active path.

    Parameters
    ----------
    allocator : QuantumRoutingAllocatorWithDilution
        The allocator instance containing the routing plan to display.
    """
    # Ensure fidelities are up-to-date before printing - just to be sure
    allocator.update_path_fidelities()

    print("\n--- Routing Plan Statistics ---")
    print(f"{allocator.get_average_fidelity()}")
    headers = ["Path (Vertex Sequence)", "Load", "F_eff"]

    if not allocator.routing_plan:
        print("Routing plan is empty.")
        return

    for idx, req in enumerate(allocator.routing_plan):
        source = req.get("source", "N/A")
        target = req.get("target", "N/A")
        total_load_req = req.get("total_load", "N/A")  # Original demand
        paths = req.get("paths", [])
        path_idxs = req.get("gid", [])  # global path IDs
        loads = req.get("path_load", [])
        fids = req.get("path_werner_param", [])

        # worst loaded path
        worst_idx = min(
            (idx for idx, path_load in enumerate(loads) if path_load > 0),
            key=fids.__getitem__,
        )

        rows = []
        actual_routed_load = 0
        # Ensure lists are available and match length needed
        min_len = min(len(paths), len(loads), len(fids))

        for i in range(min_len):
            path = paths[i]
            load = loads[i]
            fid = fids[i]

            if load >= 0:
                actual_routed_load += load
                # Construct vertex sequence string from the edge list path
                if path:  # If path is not empty
                    # Assumes path is a list of graph-tool Edge objects
                    try:
                        start_node = int(path[0].source())
                        # Get target nodes from all edges in the path
                        target_nodes = [int(e.target()) for e in path]
                        vtx_seq = "-".join(map(str, [start_node] + target_nodes))
                    except (AttributeError, IndexError):
                        vtx_seq = "Error processing path"  # Fallback
                else:
                    vtx_seq = "N/A (Empty Path)"
                if (
                    load > 0
                    or (load == 0 and show_empty_paths and not simulate_nonempty_path)
                    or (load == 0 and show_empty_paths and fid == 0)
                ):
                    rows.append(
                        (vtx_seq, load, f"{fidelity_from_werner_param(fid):.5f}")
                    )
                elif load == 0 and show_empty_paths and simulate_nonempty_path:
                    # get path index with lowest fidelity but nonzero load
                    _, trgt_multiplier = allocator.simulate_path_load_change(
                        [path_idxs[worst_idx], path_idxs[i]], 1
                    )  # TODO i don't know if I can use path_idxs[i] here.
                    rows.append(
                        (
                            vtx_seq,
                            load,
                            f"{fidelity_from_werner_param(fid*trgt_multiplier):.5f}",
                        )
                    )

        rows.sort(key=lambda x: float(x[2]), reverse=True)  # descending order

        # Print table for this request if it has any active paths
        if rows:
            print(
                f"\n=== Demand {idx}: {source} -> {target} (Requested M={total_load_req}, Routed M={actual_routed_load}) ==="
            )
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        elif total_load_req > 0:
            print(
                f"\n=== Demand {idx}: {source} -> {target} (Requested M={total_load_req}, Routed M=0) ==="
            )
            print("(No active paths or load is zero)")
        # Else: If requested M=0, maybe don't print anything unless verbose.

    print("-------------------------------\n")
