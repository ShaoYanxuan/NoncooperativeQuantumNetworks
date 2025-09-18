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
#from qroute.optimize.hill_climb import maximize_average_fidelity_greedy
from qroute.routing.allocator import QuantumRoutingAllocator, _make_edge_key
from qroute.routing.entanglement_manipulation import (
    fidelity_from_werner_param,
    werner_param_from_fidelity,
)
import graph_tool.all as gt
from qroute.routing.graph_utils import plot_allocator_network
from qroute.utils.storage import open_graph_as_flow_allocator


graph_path = '../networks/8node_example.gt.gz'; CAP=1000; f0=0.95; N=8
pair_sets = list(combinations(range(N), 2))
# F0_GRID = np.arange(0.55,0.99, 0.025)
reduced_capacity = 0
with open('./8node_example_WE_global_convex.csv', 'w') as csvfile:
        csv.writer(csvfile).writerow(['F0', '(s,t)', 'OPT', 'WE'])
# for f0 in F0_GRID:
alloc = open_graph_as_flow_allocator(graph_path,0.95)
# alloc.set_impure_edges(werner_param_from_fidelity(f0))
for s,t in pair_sets:
    print(s,t)
    min_out_deg = min(
        alloc.graph.vertex(s).out_degree(),
        alloc.graph.vertex(t).out_degree(),
    )
    demand = int(min_out_deg)
    demand_list=[(s, t, demand)]
    alloc.set_demands(demand_list)

    results = alloc.optimize_flow(nash_flow=False, print_path_fidelities=True, return_details=True)
    f_opt = alloc.compute_average_weighted_fidelity()

    results = alloc.optimize_flow(nash_flow=True, print_path_fidelities=True, return_details=True)
    f_we = alloc.compute_average_weighted_fidelity()


    print((s,t), f0, f_opt, f_we)

    with open('./8node_example_WE_global_convex.csv', 'a') as csvfile:
        csv.writer(csvfile).writerow([float(f0),(s,t),float(f_opt),float(f_we)])

