import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize
from graph_tool.all import all_paths

from qroute.routing.allocator import _make_edge_key
from qroute.routing.entanglement_manipulation import fidelity_from_werner_param
from qroute.routing.non_atomic_entanglement import (
    evaluate_convex_cost,
    integrate_convex_cost_analytical,
    integrate_convex_cost_numerical,
)
from scipy.optimize import basinhopping

TrafficRequest = Tuple[int, int, int]
EdgeIndex = Tuple[int, int]
MIN_FLOW = 1e-11
EPS_DEFAULT = 1e-10
FTOL_DEFAULT = 1e-10


class FlowAllocator:
    def __init__(
        self,
        graph,
        edge_werner_param,
        edge_capacity,
        traffic_requests,
        FTOL=FTOL_DEFAULT,
        EPS=EPS_DEFAULT,
    ):
        self.graph = graph
        self.edge_p0 = edge_werner_param
        self.edge_capacity = edge_capacity
        self.traffic_requests = traffic_requests
        self.edges = list({_make_edge_key(e) for e in graph.edges()})
        self.FTOL = FTOL
        self.EPS = EPS
        self._rebuild_paths_and_mappings()

    def _rebuild_paths_and_mappings(self):
        self.P, self.P_i, self.r = [], [], []
        for i, (src, tgt, demand) in enumerate(self.traffic_requests):
            paths = list(
                all_paths(
                    self.graph,
                    self.graph.vertex(src),
                    self.graph.vertex(tgt),
                    edges=True,
                )
            )
            self.r.append(demand)
            indices = []
            for path in paths:
                edge_list = [_make_edge_key(e) for e in path]
                indices.append(len(self.P))
                self.P.append(edge_list)
            self.P_i.append(indices)

        self.edge_to_paths = {e: [] for e in self.edges}
        for j, path in enumerate(self.P):
            for e in path:
                self.edge_to_paths[e].append(j)

    def compute_edge_flows(self, f_P):
        f_e = {e: 0 for e in self.edges}
        for j, path in enumerate(self.P):
            for e in path:
                f_e[e] += f_P[j]
        return f_e

    def _compute_cost(self, f_e, func):
        return sum(
            func(max(f_e[e], MIN_FLOW), self.edge_p0[e], self.edge_capacity[e])
            for e in self.edges
        )

    def objective(self, f_P):
        return self._compute_cost(self.compute_edge_flows(f_P), evaluate_convex_cost)

    def nash_objective(self, f_P):
        return self._compute_cost(
            self.compute_edge_flows(f_P), integrate_convex_cost_analytical
        )

    # def nash_objective(self, f_P):
    #     return self._compute_cost(
    #         self.compute_edge_flows(f_P), integrate_convex_cost_numerical
    #     )

    def constraint_flow_conservation(self, f_P):
        return np.array(
            [
                sum(f_P[j] for j in self.P_i[i]) - demand
                for i, demand in enumerate(self.r)
            ]
        )

    def optimize_flow(
        self, nash_flow=False, print_path_fidelities=False, return_details=False
    ):
        num_paths = len(self.P)
        bounds = [(0, None)] * num_paths
        result = minimize(
            self.nash_objective if nash_flow else self.objective,
            x0=np.ones(num_paths) * 0.1,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": self.constraint_flow_conservation}],
            options={
                "disp": False,
                "ftol": self.FTOL,
                "eps": self.EPS,
                "maxiter": 1_000,
            },
        )

        if not result.success:
            raise RuntimeError(f"Optimizer failed: {result.message}")

        self.f_P_opt = result.x

        if print_path_fidelities:
            self._print_path_fidelities()

        path_costs = self._compute_path_costs(result.x)

        avg_fid = self.compute_average_weighted_fidelity()
        if return_details:
            return {
                "average_weighted_fidelity": avg_fid,
                "optimal_path_flows": result.x,
                "path_costs": path_costs,
            }
        else:
            return avg_fid

    def optimize_flow_NEW(
        self,
        nash_flow=False,
        print_path_fidelities=False,
        return_details=False,
        excluded_path_indices=None,
    ):
        num_paths = len(self.P)
        if excluded_path_indices is None:
            excluded_path_indices = []

        # Build bounds: force excluded paths to 0
        bounds = [
            (0, 0) if i in excluded_path_indices else (0, None)
            for i in range(num_paths)
        ]

        # Build initial guess, zero for excluded paths
        x0 = np.ones(num_paths) * 0.3
        for i in excluded_path_indices:
            x0[i] = 0.0

        result = minimize(
            self.nash_objective if nash_flow else self.objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": self.constraint_flow_conservation}],
            options={
                "disp": False,
                "ftol": self.FTOL,
                "eps": self.EPS,
                "maxiter": 1_000,
            },
        )

        if not result.success:
            raise RuntimeError(f"Optimizer failed: {result.message}")

        self.f_P_opt = result.x

        if print_path_fidelities:
            self._print_path_fidelities()

        path_costs = self._compute_path_costs(result.x)
        avg_fid = self.compute_average_weighted_fidelity()

        if return_details:
            return {
                "average_weighted_fidelity": avg_fid,
                "optimal_path_flows": result.x,
                "path_costs": path_costs,
            }
        else:
            return avg_fid

    def _reconstruct_node_sequence(self, edge_list, src):
        if not edge_list:
            return ""
        u, v = edge_list[0]
        ordered = edge_list if u == src else [(v, u) for (u, v) in edge_list]
        seq = [ordered[0][0], ordered[0][1]]
        for e in ordered[1:]:
            seq.append(e[1] if e[0] == seq[-1] else e[0])
        return "-".join(map(str, seq))

    def set_demands(self, new_demands: List[TrafficRequest]):
        self.traffic_requests = new_demands
        self._rebuild_paths_and_mappings()

    def set_impure_edges(self, new_p: float):
        for e in self.edge_p0:
            if abs(self.edge_p0[e] - 1.0) > 1e-10:
                self.edge_p0[e] = new_p

    def set_edge_capacity(self, source: int, target: int, new_capacity: int):
        edge_key = (source, target)
        if edge_key not in self.edges:
            raise ValueError(f"Edge {edge_key} not found.")
        self.edge_capacity[edge_key] = new_capacity
        self._rebuild_paths_and_mappings()

    def optimize_better_than_wardrop(
        self, return_details=False, print_path_fidelities=False
    ):
        """
        Optimize under the constraint that no user has higher cost than in the Wardrop equilibrium.
        """
        wardrop_details = self.optimize_flow(nash_flow=True, return_details=True)
        wardrop_costs = wardrop_details["path_costs"]
        f_P_WEs = wardrop_details["optimal_path_flows"]

        def constraint_better_than_wardrop(f_Ps):
            f_P_WEs = wardrop_details["optimal_path_flows"]
            new_costs = self._compute_path_costs(f_Ps)

            # Only consider used paths in both Wardrop and new flow
            used_old_costs = [c for c, f in zip(wardrop_costs, f_P_WEs) if f > MIN_FLOW]
            used_new_costs = [c for c, f in zip(new_costs, f_Ps) if f > MIN_FLOW]

            # viol = [new_costs[j] - wardrop_costs[j] for j in range(len(f_Ps))]
            viol = np.zeros_like(f_Ps)
            for j in range(len(f_Ps)):
                if f_Ps[j] > MIN_FLOW or f_P_WEs[j] > MIN_FLOW:
                    viol[j] = new_costs[j] - wardrop_costs[j]  # TODO
                # if f_Ps[j] > MIN_FLOW:
                #     viol[j] = new_costs[j] - 0.6  # TODO
                else:
                    viol[j] = 0  # unused paths → no constraint violation

            return viol

            # Constraint: new >= old  →  new-old  ≥ 0
            return np.array(viol)

        num_paths = len(self.P)
        result = minimize(
            self.objective,
            x0=f_P_WEs,
            # x0=np.ones(num_paths) * 0.1,
            method="SLSQP",
            bounds=[(0, None)] * num_paths,
            constraints=[
                {"type": "eq", "fun": self.constraint_flow_conservation},
                {"type": "ineq", "fun": constraint_better_than_wardrop},
            ],
            options={"ftol": self.FTOL, "eps": self.EPS, "maxiter": 1000},
        )

        if not result.success:
            raise RuntimeError(
                f"Better-than-Wardrop optimization failed: {result.message}"
            )

        self.f_P_opt = result.x
        if print_path_fidelities:
            self._print_path_fidelities()
            print(self._compute_path_costs(result.x))
            print(constraint_better_than_wardrop(result.x))
            # exit()
        return self.compute_average_weighted_fidelity()

    from scipy.optimize import basinhopping

    def optimize_better_than_wardrop_basin(
        self, return_details=False, print_path_fidelities=False
    ):
        wardrop_details = self.optimize_flow(nash_flow=True, return_details=True)
        f_P_WEs = wardrop_details["optimal_path_flows"]
        wardrop_costs = wardrop_details["path_costs"]

        def constraint_better_than_wardrop(f_Ps):
            new_costs = self._compute_path_costs(f_Ps)
            viol = np.zeros_like(f_Ps)
            for j in range(len(f_Ps)):
                if f_Ps[j] > MIN_FLOW:
                    viol[j] = 0.5 - new_costs[j]
                else:
                    viol[j] = 0
            return viol

        constraints = [
            {"type": "eq", "fun": self.constraint_flow_conservation},
            {"type": "ineq", "fun": constraint_better_than_wardrop},
        ]

        bounds = [(0, None)] * len(self.P)

        def local_minimizer(x0):
            return minimize(
                self.objective,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": self.FTOL, "eps": self.EPS, "maxiter": 1000},
            )

        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints,
            "options": {"ftol": self.FTOL, "eps": self.EPS, "maxiter": 1000},
        }

        result = basinhopping(
            self.objective,
            x0=f_P_WEs,  # or random vector
            minimizer_kwargs=minimizer_kwargs,
            niter=5,  # you can try higher
            disp=True,
        )

        if not result.lowest_optimization_result.success:
            raise RuntimeError(f"Basin hopping failed: {result.message}")

        self.f_P_opt = result.x
        if print_path_fidelities:
            self._print_path_fidelities()

        return self.compute_average_weighted_fidelity()

    def _compute_path_costs(self, f_P):
        f_e = self.compute_edge_flows(f_P)
        costs = []
        for path in self.P:
            total = 0
            for e in path:
                fv = max(f_e[e], MIN_FLOW)
                total += evaluate_convex_cost(
                    fv, self.edge_p0[e], self.edge_capacity[e]
                )
            costs.append(total)

        path_details = []
        avg_fidelity = 0
        for j, path in enumerate(self.P):
            flow = self.f_P_opt[j]
            fidelity = None

            prod = 1
            for e in path:
                val = max(f_e[e], MIN_FLOW)
                cost = evaluate_convex_cost(val, self.edge_p0[e], self.edge_capacity[e])
                prod *= np.exp(-cost / val)
            fidelity = fidelity_from_werner_param(prod)
            avg_fidelity += flow * fidelity
            path_details.append(fidelity)
        # return costs
        return costs

    def compute_average_weighted_fidelity(self):
        if not hasattr(self, "f_P_opt"):
            raise ValueError("No optimal flow found yet.")
        f_e_final = self.compute_edge_flows(self.f_P_opt)
        total_demand = sum(self.r)
        avg_fidelity = 0

        for j, path in enumerate(self.P):
            flow = self.f_P_opt[j]
            if flow > MIN_FLOW:
                prod = 1
                for e in path:
                    val = max(f_e_final[e], MIN_FLOW)
                    cost = evaluate_convex_cost(
                        val, self.edge_p0[e], self.edge_capacity[e]
                    )
                    prod *= np.exp(-cost / val)
                fidelity = fidelity_from_werner_param(prod)
                avg_fidelity += flow * fidelity

        return avg_fidelity / total_demand

    def _print_path_fidelities(self):
        # assumes self.f_P_opt is set
        f_e_final = self.compute_edge_flows(self.f_P_opt)
        total_demand = sum(self.r)

        path_details = []
        avg_fidelity = 0
        for j, path in enumerate(self.P):
            flow = self.f_P_opt[j]
            fidelity = None
            if flow > MIN_FLOW:
                prod = 1
                for e in path:
                    val = max(f_e_final[e], MIN_FLOW)
                    cost = evaluate_convex_cost(
                        val, self.edge_p0[e], self.edge_capacity[e]
                    )
                    prod *= np.exp(-cost / val)
                fidelity = fidelity_from_werner_param(prod)
                avg_fidelity += flow * fidelity
            path_details.append(
                {"index": j, "flow": flow, "fidelity": fidelity, "edges": path}
            )

        for idx, (src, tgt, demand) in enumerate(self.traffic_requests):
            print(f"\nRequest {idx}: (src={src} → tgt={tgt}, demand={demand})")
            print(f"{'Path':>30} {'Flow':>10} {'Fidelity':>12}")
            print("-" * 60)
            for p in sorted(
                (path_details[j] for j in self.P_i[idx]), key=lambda x: -x["flow"]
            ):
                flow_str = f"{p['flow']:.9f}"
                fid_str = f"{p['fidelity']:.6f}" if p["fidelity"] else "unused"
                node_seq = self._reconstruct_node_sequence(p["edges"], src)
                print(f"{node_seq:>30} {flow_str:>10} {fid_str:>12}")
        print(f"\nAverage weighted fidelity: {avg_fidelity/total_demand:.6f}")

    def get_min_degree(self, source, target):
        return min(self.graph.get_out_degrees([source, target]))


def remove_edges_and_rebuild_allocator(allocator, edges_to_remove):
    remove_set = set(edges_to_remove)
    g_copy = allocator.graph.copy()

    for e in list(g_copy.edges()):
        u, v = int(e.source()), int(e.target())
        if (u, v) in remove_set or (v, u) in remove_set:
            g_copy.remove_edge(e)

    new_edge_p0 = {
        e: p
        for e, p in allocator.edge_p0.items()
        if e not in remove_set and (e[1], e[0]) not in remove_set
    }
    new_edge_cap = {
        e: c
        for e, c in allocator.edge_capacity.items()
        if e not in remove_set and (e[1], e[0]) not in remove_set
    }

    return FlowAllocator(g_copy, new_edge_p0, new_edge_cap, allocator.traffic_requests)
