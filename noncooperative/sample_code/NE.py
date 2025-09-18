import sys
sys.path.append('../')
from itertools import combinations
import multiprocessing as mp
from qroute.routing.utils import print_routing_plan_statistics
import numpy as np
import csv
from datetime import datetime
# ---------------

import matplotlib.pyplot as plt
from graph_tool import Graph
from qroute.optimize.selfish_routing import find_discrete_nash_equilibrium
# from qroute.optimize.hill_climb import maximize_average_fidelity_greedy
from qroute.routing.allocator import QuantumRoutingAllocator, _make_edge_key
from qroute.routing.entanglement_manipulation import (
    fidelity_from_werner_param,
    werner_param_from_fidelity,
)
from qroute.utils.storage import (
    open_graph_as_flow_allocator,
    open_graph_as_quantum_allocator,
)
import graph_tool.all as gt


def run_one_graph(graph_path, reduced_capacity=0):
    pair_sets = list(combinations(range(N), 2))
    # graph = gt.load_graph(graph_path)
    alloc = open_graph_as_quantum_allocator(graph_path, 0.55, CAP)
    with open('./8node_example_NE.csv', 'w') as csvfile:
        csv.writer(csvfile).writerow(['F0', '(s,t)', 'NE', 'NE after removal'])

    #F_before_after = []
    for f0 in F0_grid:
        print(datetime.now())
        alloc.set_impure_edges(werner_param_from_fidelity(f0))

        f_before_after = []
        for s,t in pair_sets:
            min_out_deg = min(
                alloc.graph.vertex(s).out_degree(),
                alloc.graph.vertex(t).out_degree(),
            )
            demand = int(min_out_deg * CAP)
            demand_list=[(s, t, demand)]
            alloc.set_demands(demand_list)
            f_before = find_discrete_nash_equilibrium(alloc, load_shift_list=[1, 5, 25, 125])

            highest_f_after = f_before
            for u,v in alloc.edges:
                alloc.set_edge_capacity(u,v,reduced_capacity)
                f_after = find_discrete_nash_equilibrium(alloc)
                alloc.set_edge_capacity(u, v, CAP)
                highest_f_after = max(highest_f_after, f_after)
            print(s,t,f0,float(f_before),float(highest_f_after))
            with open('./8node_example_NE.csv', 'a') as csvfile:
                csv.writer(csvfile).writerow([float(f0),(s,t),float(f_before),float(highest_f_after)])
    return


if __name__ == "__main__":
    #Check out experiments.combination.multi_user_capacity
    N = 8
    CAP = 1000
    F0_grid = np.arange(0.55,0.999,0.025)
    print("RUNNING")
    graph_path = '../networks/8node_example.gt.gz'
    run_one_graph(graph_path)



