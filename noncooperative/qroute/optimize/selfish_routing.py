# qroute/optimize/selfish_routing.py


def find_discrete_nash_equilibrium(
    alloc, max_iters=1_000, tol: float = 1e-6, load_shift_list=[1, 4, 16]
) -> float:
    """
    Runs discrete Nash equilibrium search.

    The routine sweeps over each demand and repeatedly shifts single
    load units from a loaded path with lower fidelity to a better
    one, as long as that move strictly improves fidelity by > `tol`.

    Returns
    -------
    float
        The average fidelity of the final routing plan.
    """
    load_shift = load_shift_list[-1]

    for _ in range(max_iters):
        moved_any = False  # becomes True if any demand improves this sweep

        for demand in alloc.routing_plan:
            path_loads = demand["path_load"]
            if len(path_loads) < 2 or not any(path_loads):
                continue  # single-path demand or nothing to move

            path_ids = demand["gid"]  # global path IDs
            path_wp = demand["path_werner_param"]  # alias - doesn't need to be updated
            get_wp = path_wp.__getitem__
            num_paths = len(path_loads)

            # keep cycling until no move helps this demand
            improved = True
            while improved:
                improved = False

                # worst-to-best list of currently loaded paths
                loaded_path_idx = sorted(
                    (idx for idx, path_load in enumerate(path_loads) if path_load > 0),
                    key=get_wp,
                )

                # index of best path
                best_idx = max(range(num_paths), key=get_wp)

                # try moving load off each loaded path, starting with the worst
                for src_idx in loaded_path_idx:
                    # Early exit if best path isn't better than current one
                    if path_wp[best_idx] <= path_wp[src_idx] + tol:
                        break

                    # Special case: src fidelity zero (if capacity=0)
                    if path_wp[src_idx] == 0.0:
                        delta = path_loads[src_idx]
                        alloc.update_path_fidelities_specific(
                            [path_ids[src_idx], path_ids[best_idx]], delta
                        )
                        path_loads[src_idx] -= delta
                        path_loads[best_idx] += delta
                        moved_any = improved = True
                        continue  # Move all to the path with the best fidelity and continue to next src

                    # candidate destinations, strictly better than current path
                    trgt_idxs = [
                        i
                        for i in range(num_paths)
                        if i != src_idx and path_wp[i] > path_wp[src_idx] + tol
                    ]
                    trgt_idxs.sort(key=get_wp, reverse=True)

                    for trgt_idx in trgt_idxs:
                        if path_loads[src_idx] < load_shift:
                            break
                        # simulate moving unit src → dest
                        changed = [path_ids[src_idx], path_ids[trgt_idx]]
                        _, trgt_multiplier = alloc.simulate_path_load_change(
                            changed, load_shift
                        )

                        while (
                            path_loads[src_idx] >= load_shift
                            and trgt_multiplier * path_wp[trgt_idx]
                            > path_wp[src_idx] + tol  # new path better
                        ):
                            alloc.update_path_fidelities_specific(changed, load_shift)
                            path_loads[src_idx] -= load_shift
                            path_loads[trgt_idx] += load_shift
                            moved_any = improved = True

                            if path_loads[src_idx] < load_shift:
                                break
                            _, trgt_multiplier = alloc.simulate_path_load_change(
                                changed, load_shift
                            )
                    if improved:
                        break  # recompute loaded_idxs and try again
        if not moved_any:
            break  # full sweep without improvements → converged
    else:
        print(f"Warning: max_iters={max_iters} reached without convergence")

    if len(load_shift_list) > 1:
        return find_discrete_nash_equilibrium(
            alloc, max_iters, tol, load_shift_list[:-1]
        )
    else:
        return alloc.get_average_fidelity()
