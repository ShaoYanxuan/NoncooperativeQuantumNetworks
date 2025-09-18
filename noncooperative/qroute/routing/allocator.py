from collections import Counter
import copy
import random
from typing import List, Tuple, Dict, Optional
from graph_tool.all import Graph, PropertyMap, all_paths

from qroute.routing.entanglement_manipulation import (
    calculate_effective_edge_p,
    fidelity_from_werner_param,
)

TrafficRequest = Tuple[int, int, int]  # (source, target, demand)
EdgeIndex = Tuple[int, int]  # (source, target)
WERNER_PARAM_THRESHOLD = 1e-6
max_path_length = None
max_split_paths = 3


def _make_edge_key(e):
    s, t = int(e.source()), int(e.target())
    return (s, t) if s < t else (t, s)


class QuantumRoutingAllocator:
    """
    Quantum routing allocator with edge dilution and purification logic.

    If M > N, Werner parameter is diluted.
    If M < N, purification is performed using BBPSSW.
    """

    def __init__(
        self,
        graph: Graph,
        edge_werner_param: Dict[EdgeIndex, float],
        edge_capacity: Dict[EdgeIndex, int],
        traffic_requests: List[TrafficRequest],
        shots=1_000,  # had 1_000_000
        precompute_cache=True,
    ):
        # ----- store static inputs -----------------------------------------
        self.graph = graph
        self.shots = shots
        self.precompute_cache = precompute_cache
        self.traffic_requests = traffic_requests  # (src, dst, demand)
        self.max_requested_states = sum(t[2] for t in traffic_requests) + 1

        self.edge_keys = {e: _make_edge_key(e) for e in graph.edges()}
        self.edges = [self.edge_keys[e] for e in graph.edges()]

        # Edge properties in plain dicts (fast, hash-map look-ups)
        self.edge_p0 = {}  # Werner parameter  p₀
        self.edge_capacity = {}  # Raw Bell pairs   N
        for k in self.edges:
            self.edge_p0[k] = float(edge_werner_param[k])
            self.edge_capacity[k] = int(edge_capacity[k])

        # Mutable per-edge demand M (initially zero)
        self.edge_demand = {k: 0 for k in self.edges}

        # ----- routing-state scaffolding -----------------------------------
        self.routing_plan = []  # list[dict] per request
        self.all_paths_flat = []  # list[list[Tuple[int,int]]]
        self.path_index_to_request = []  # global → (req_idx, path_idx)
        self.path_edges = []  # cached edge keys per path
        self.path_overlap = []  # list[set[int]]   (optional)
        self.edge_incidence = {k: [] for k in self.edges}
        self.path_effective_p = []  # list[float]

        self._peff_lut = {
            k: [
                calculate_effective_edge_p(
                    self.edge_p0[k],
                    self.edge_capacity[k],
                    m,
                    self.shots,
                    self.precompute_cache,
                )
                for m in range(self.max_requested_states + 1)
            ]
            for k in self.edges
        }

        # Initialise everything derived from inputs
        self.initialize_routing()

    def initialize_routing(
        self, max_path_length: int | None = None, max_split_paths: int = 3
    ) -> None:
        """Find paths for every traffic request and prime all per-path caches."""
        # ---------- reset global state ----------------------------------------
        self.routing_plan.clear()
        self.all_paths_flat = []
        self.path_index_to_request = []
        self.path_edges = []  # cached tuple keys per path
        self.path_edge_set = []
        self.edge_demand.update({k: 0 for k in self.edge_demand})  # zero M

        # ---------- per-request processing -----------------------------------
        for req_idx, (src, tgt, demand) in enumerate(self.traffic_requests):
            src_v, tgt_v = self.graph.vertex(src), self.graph.vertex(tgt)

            # all simple paths (edge lists) ------------------------------------
            paths = list(
                all_paths(self.graph, src_v, tgt_v, cutoff=max_path_length, edges=True)
            )
            paths.sort(key=len)  # shorter first
            if not paths:
                self.routing_plan.append(
                    {  # keep placeholder entry
                        "source": src,
                        "target": tgt,
                        "total_load": demand,
                        "paths": [],
                        "path_load": [],
                        "path_werner_param": [],
                        "gid": [],
                    }
                )
                continue

            # distribute demand across k paths --------------------------------
            k = min(len(paths), max_split_paths, max(demand, 1))
            path_load = [0] * len(paths)
            for u in range(demand):
                path_load[u % k] += 1  # round-robin

            # global path bookkeeping -----------------------------------------
            start_pid = len(self.all_paths_flat)
            self.all_paths_flat.extend(paths)
            self.path_index_to_request.extend((req_idx, j) for j in range(len(paths)))
            gids = list(range(start_pid, start_pid + len(paths)))

            # cache tuple-key edge lists for speed
            edges_for_paths = [[self.edge_keys[e] for e in p] for p in paths]
            self.path_edges.extend(edges_for_paths)
            self.path_edge_set.extend(frozenset(pe) for pe in edges_for_paths)

            # store request meta-data
            self.routing_plan.append(
                {
                    "source": src,
                    "target": tgt,
                    "total_load": demand,
                    "paths": paths,
                    "path_load": path_load,
                    "path_werner_param": [0.0] * len(paths),
                    "gid": gids,
                }
            )

        # ---------- build overlaps & edge incidence ---------------------------
        n_paths = len(self.all_paths_flat)
        self.path_overlap = [set() for _ in range(n_paths)]
        self.edge_incidence = {k: [] for k in self.edge_demand}

        for pid, edge_keys in enumerate(self.path_edges):
            ek_set = set(edge_keys)
            # edge incidence
            for k in edge_keys:
                self.edge_incidence[k].append(pid)
            # overlaps (naïve O(P²) scan is okay for ≤ few k·requests)
            for pj in range(pid + 1, n_paths):
                if ek_set & set(self.path_edges[pj]):
                    self.path_overlap[pid].add(pj)
                    self.path_overlap[pj].add(pid)

        # ---------- initial fidelity calc -------------------------------------
        self.update_path_fidelities()

    def sample_random_routing_plan(self, *, skew: float = 2.0):
        """
        Randomly assign the path loads for each request using a skewed distribution.

        This function fully resets the routing plan by assigning each request's load
        randomly across its available paths. The `skew` parameter controls how uneven
        the random distribution is: higher values increase the likelihood that one path
        receives most of the load.

        This is useful for random-restart optimization (e.g. shotgun hill climbing),
        where multiple initial routing plans are evaluated.

        Parameters
        ----------
        skew : float, optional
            Exponent applied to the random weights. Larger values increase skew toward
            a single dominant path. Default is 2.0.
        """
        for route_info in self.routing_plan:
            total_load = route_info["total_load"]
            num_paths = len(route_info["paths"])

            if total_load == 0 or num_paths < 2:
                continue

            base_weights = [random.random() + 1e-9 for _ in range(num_paths)]
            weights = [w**skew for w in base_weights]

            choices = random.choices(range(num_paths), weights=weights, k=total_load)
            counts = Counter(choices)

            route_info["path_load"] = [counts.get(i, 0) for i in range(num_paths)]

        self.update_path_fidelities()

    # ------------------------------------------------------------------
    #  FIDELITY UPDATES
    # ------------------------------------------------------------------
    def update_path_fidelities(self) -> None:
        """
        1) Re-compute demand M for every edge.
        2) Update self.edge_effective_p[k]  (k = (u,v) tuple).
        3) Update per-path Wer­ner parameters inside each request.
        """
        # ---------- 1) edge-demand reset & accumulation ----------------------
        for k in self.edge_demand:
            self.edge_demand[k] = 0

        for req in self.routing_plan:
            for path, load in zip(req["paths"], req["path_load"]):
                if load == 0:
                    continue
                for e in path:
                    self.edge_demand[self.edge_keys[e]] += load

        # ---------- 2) edge-effective-p update --------------------------------
        self.edge_effective_p = {}  # (u,v) → p_eff
        for k in self.edge_p0:
            self.edge_effective_p[k] = self._peff_lut[k][self.edge_demand[k]]

        # ---------- 3) per-path fidelities -----------------------------------
        for req in self.routing_plan:
            fids = []
            for path in req["paths"]:
                if not path:
                    fids.append(0.0)
                    continue

                fid = 1.0
                for e in path:
                    fid *= self.edge_effective_p[self.edge_keys[e]]
                    if fid <= WERNER_PARAM_THRESHOLD:
                        fid = 0.0
                        break
                fids.append(fid)

            req["path_werner_param"] = fids

    def simulate_path_load_change(
        self, changed_path_ids: tuple[int, int], delta=1, need_source_multiplier=False
    ):

        # Get edge sets for the source (load decreased) and target (load increased) paths
        source_path_edges = self.path_edge_set[changed_path_ids[0]]
        target_path_edges = self.path_edge_set[changed_path_ids[1]]

        # Determine unique edges used by each path (no double-counting shared edges)
        only_source_edges = source_path_edges - target_path_edges
        only_target_edges = target_path_edges - source_path_edges

        if need_source_multiplier:
            source_path_multiplier = 1.0
            for edge_key in only_source_edges:
                source_path_multiplier *= (
                    self._peff_lut[edge_key][self.edge_demand[edge_key] - delta]
                    / self.edge_effective_p[edge_key]
                )
        else:
            source_path_multiplier = None

        target_path_multiplier = 1.0
        for edge_key in only_target_edges:
            target_path_multiplier *= (
                self._peff_lut[edge_key][self.edge_demand[edge_key] + delta]
                / self.edge_effective_p[edge_key]
            )

        return source_path_multiplier, target_path_multiplier

    def update_path_fidelities_specific(
        self, changed_path_ids: tuple[int, int], delta=1
    ) -> None:
        """
        Incrementally refresh edge demand M, edge-effective p, and path fidelities
        when a pair of paths has changed load.

        Parameters
        ----------
        changed_path_ids : tuple[int, int]
            A pair of global path IDs whose load counters were just modified.
        delta : int, optional
            Amount to adjust the path loads by. First ID gets -delta, second gets +delta.
        """
        # Get edge sets for the source (load decreased) and target (load increased) paths
        source_path_edges = self.path_edge_set[changed_path_ids[0]]
        target_path_edges = self.path_edge_set[changed_path_ids[1]]

        # Determine unique edges used by each path (no double-counting shared edges)
        only_source_edges = source_path_edges - target_path_edges
        only_target_edges = target_path_edges - source_path_edges

        # Update edges affected by the source path (demand decreases)
        for edge_key in only_source_edges:
            self.edge_demand[edge_key] -= delta
            self.edge_effective_p[edge_key] = self._peff_lut[edge_key][
                self.edge_demand[edge_key]
            ]

        # Update edges affected by the target path (demand increases)
        for edge_key in only_target_edges:
            self.edge_demand[edge_key] += delta
            self.edge_effective_p[edge_key] = self._peff_lut[edge_key][
                self.edge_demand[edge_key]
            ]

        # Recompute werner params for all paths directly or indirectly affected
        affected_paths = set(changed_path_ids).union(
            *[self.path_overlap[pid] for pid in changed_path_ids]
        )
        # TODO maybe could be improved by only working with the specific edges
        for path_idx in affected_paths:
            req_idx, loc_idx = self.path_index_to_request[path_idx]
            new_werner_param = 1.0
            for edge_key in self.path_edges[path_idx]:
                new_werner_param *= self.edge_effective_p[edge_key]
                if new_werner_param <= WERNER_PARAM_THRESHOLD:
                    new_werner_param = 0.0
                    break
            self.routing_plan[req_idx]["path_werner_param"][loc_idx] = new_werner_param

    # ------------------------------------------------------------------
    #  METRICS
    # ------------------------------------------------------------------
    def get_average_werner_param(self) -> float:
        """Σ(load × fid) / Σ load over *all* paths with positive load."""
        tot_load, tot_weighted = 0, 0.0
        for req in self.routing_plan:
            for load, fid in zip(req["path_load"], req["path_werner_param"]):
                if load:
                    tot_load += load
                    tot_weighted += load * fid
        return tot_weighted / tot_load if tot_load else 0.0

    def get_average_fidelity(self) -> float:
        return fidelity_from_werner_param(self.get_average_werner_param())

    def get_average_werner_param_per_demand(self) -> Dict[Tuple[int, int], float]:
        """
        (src, tgt) ↦ Σ(load × fid) / Σ load for that request.
        Only returns entries with non-zero load.
        """
        out: Dict[Tuple[int, int], float] = {}
        for req in self.routing_plan:
            total_load = sum(req["path_load"])
            if total_load:
                weighted = sum(
                    l * f for l, f in zip(req["path_load"], req["path_werner_param"])
                )
                out[(req["source"], req["target"])] = weighted / total_load
        return out

    # ------------------------------------------------------------------
    #  STATE MODIFICATION
    # ------------------------------------------------------------------
    def set_demands(self, new_demands: list[TrafficRequest]) -> None:
        """Replace current traffic requests and rebuild the routing plan."""
        self.traffic_requests = new_demands
        self.max_requested_states = sum(t[2] for t in new_demands) + 1

        # Extend each edge LUT if needed
        for k in self.edges:
            current_len = len(self._peff_lut[k])
            if current_len < self.max_requested_states + 1:
                self._peff_lut[k].extend(
                    [
                        calculate_effective_edge_p(
                            self.edge_p0[k],
                            self.edge_capacity[k],
                            m,
                            self.shots,
                            self.precompute_cache,
                        )
                        for m in range(current_len, self.max_requested_states + 1)
                    ]
                )
        self.initialize_routing()

    def set_impure_edges(self, new_p: float) -> None:
        """
        Set p₀ = new_p on edges that aren’t already (numerically) perfect.
        Perfect means |p₀ − 1| ≤ WERNER_PARAM_THRESHOLD.
        """
        for edge_key, p0 in self.edge_p0.items():
            if abs(p0 - 1.0) > WERNER_PARAM_THRESHOLD:
                self.edge_p0[edge_key] = new_p
                self._recompute_edge_lut(edge_key)
        self.update_path_fidelities()

    def set_edge_capacity(self, source: int, target: int, new_capacity: int) -> None:
        edge_key = (source, target)
        if edge_key not in self.edges:
            raise ValueError(f"Edge {edge_key} not found.")

        self.edge_capacity[edge_key] = new_capacity
        self._recompute_edge_lut(edge_key)
        self.update_path_fidelities()

    def set_edge_fidelity(self, source: int, target: int, new_p0: float):
        edge_key = (source, target)
        if edge_key not in self.edges:
            raise ValueError(f"Edge {edge_key} not found.")

        self.edge_p0[edge_key] = new_p0
        self._recompute_edge_lut(edge_key)
        self.update_path_fidelities()

    def _recompute_edge_lut(self, edge_key):
        """
        Recalculate the effective fidelity LUT for a given edge.
        Handles zero-capacity edges by filling with 0.0s.
        """
        capacity = self.edge_capacity[edge_key]

        if capacity == 0:
            self._peff_lut[edge_key] = [
                0.0 for _ in range(self.max_requested_states + 1)
            ]
        else:
            self._peff_lut[edge_key] = [
                calculate_effective_edge_p(
                    self.edge_p0[edge_key],
                    capacity,
                    m,
                    self.shots,
                    self.precompute_cache,
                )
                for m in range(self.max_requested_states + 1)
            ]

    def get_min_degree(self, source, target):
        return min(self.graph.get_out_degrees([source, target]))
