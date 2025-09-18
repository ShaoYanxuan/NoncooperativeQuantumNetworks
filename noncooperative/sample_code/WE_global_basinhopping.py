import numpy as np
import csv
import graph_tool.all as gt
import math
import random
from itertools import chain
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
from scipy.optimize import bisect
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping

def func(x, S):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)-S
def solve_sch_co(S):
    lambda0 = fsolve(func, 1e-5, args=S)[0]
    return lambda0

def Edge_count(graph, path_list, path_count_list):
    edge_count = {}
    for edge in graph.get_edges():
        edge_count[(min(edge[0], edge[1]), max(edge[0], edge[1]))] = 0
    for i in range(len(path_list)):
        path = path_list[i]
        for j in range(len(path)-1):
            edge_count[(min(path[j],path[j+1]), max(path[j],path[j+1]))] += path_count_list[i]
    return edge_count

def F_value(p):
    return (3*p+1)/4

def p_value(F):
    return (4*F-1)/3

def f1(x):
    if x<=1:
        return 1
    else:
        lambda0 = solve_sch_co(1/x)
        return (np.sqrt(lambda0*(1-lambda0))*4+1)/3

def f2(x):
    if x<1:
        p_index = int(x*M)
        return avg_mixed_p[p_index] + (avg_mixed_p[p_index+1]-avg_mixed_p[p_index])*(x*M-p_index)
    else:
        return p/x

def con(x):
    return sum(x)-m
cons = {'type':'eq', 'fun': con}

def opt_func(x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    path_p_list = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        path_p = 1
        for item in edge_index:
            if bell_mixed[item] == 0:
                path_p = path_p*f1(edge_num[item])
            else:
                path_p = path_p*f2(edge_num[item])
        path_p_list.append(path_p)
    avgP = sum([path_p_list[i]*x[i] for i in range(len(path_p_list))])/m
    equation = -avgP 
    return equation

def opt_avgP(A,path_list,edge_list,x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    pathPList = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        pathP = 1
        for item in edge_index:
            if bell_mixed[item] == 0:
                pathP = pathP*f1(edge_num[item])
            else:
                pathP = pathP*f2(edge_num[item])
        pathPList.append(pathP)
        
    avgP = sum([pathPList[i]*x[i] for i in range(len(pathPList))])/m
    equation = -avgP + (sum([x[i] for i in range(len(path_list))])-m)**2
    return avgP, equation


def theory_nash(x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    path_p_list = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        path_p = 1
        for item in edge_index:
            if bell_mixed[item] == 0:
                path_p = path_p*f1(edge_num[item])
            else:
                path_p = path_p*f2(edge_num[item])
        path_p_list.append(path_p)
    avgP = sum([path_p_list[i]*x[i] for i in range(len(path_p_list))])/m
    equation = 0
    for i in range(len(x)):
        # equation += (path_p_list[i]-avgP)**2*x[i]
        if path_p_list[i]>avgP:
            equation += (path_p_list[i]-avgP)**2
        else:
            equation += (path_p_list[i]-avgP)**2*x[i]
    return equation
    
def theory_nashP(A,path_list,edge_list,x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    pathPList = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        pathP = 1
        for item in edge_index:
            if bell_mixed[item] == 0:
                pathP = pathP*f1(edge_num[item])
            else:
                pathP = pathP*f2(edge_num[item])
        pathPList.append(pathP)
        
    avgP = sum([pathPList[i]*x[i] for i in range(len(pathPList))])/m
    equation = 0
    for i in range(len(x)):
        # equation += (pathPList[i]-avgP)**2*x[i]
        if pathPList[i]>avgP:
            equation += (pathPList[i]-avgP)**2
        else:
            equation += (pathPList[i]-avgP)**2*x[i]
    return avgP, equation

# load network ------------------------------------------------------------
g = gt.load_graph('../networks/8node_example.gt.gz')
ent = g.edge_properties["entanglement"]
ent_dict = {}
for edge in g.get_edges():
    ent_dict[(min(edge[0],edge[1]),max(edge[0],edge[1]))] = ent[g.edge(edge[0],edge[1])]
# gt.graph_draw(g, vertex_fill_color='grey', vertex_text=g.vertex_index, edge_color=ent, bg_color='lightgrey')
N = 8; M = 1000; F0 = 0.95; p = p_value(F0)
with open('../purification/purification_F_1000.csv', 'r') as csvfile:
    avg_mixed_p = [p_value(float(item)) for item in list(csv.reader(csvfile))[-1]]

with open('./8node_example_WE_global_basinhopping.csv', 'w') as csvfile:
            csv.writer(csvfile).writerow(['F0','(s,t)','OPT','WE'])
edge_list = []; bell_mixed = []
for edge in g.iter_edges():
    edge_list.append(edge)
    bell_mixed.append(ent[g.edge(edge[0],edge[1])])
for start in range(N):
    for end in range(start+1,N):
        m = min(g.vertex(start).out_degree(), g.vertex(end).out_degree())
        path_list = []
        for path in gt.all_paths(g, start, end):
            path_list.append(list(path))
        X0 = [0 for i in path_list]; X0[0] = m
        A = [[0 for i in range(len(path_list))] for j in range(len(edge_list))]
        for i in range(len(path_list)):
            path = path_list[i]
            for j in range(len(path)-1):
                edge = list(np.sort([path[j],path[j+1]]))
                if edge in edge_list:
                    A[edge_list.index(edge)][i] = 1
        lowerBounds = 0*np.ones(len(path_list))
        upperBounds = np.ones(len(path_list))*m
        boundData=Bounds(lowerBounds,upperBounds)
        kwargs = dict(bounds=boundData, constraints=cons)
        result = basinhopping(opt_func, X0, minimizer_kwargs=kwargs)
        x = result.x
        avgP_opt, minimization = opt_avgP(A,path_list,edge_list,x)

        result = basinhopping(theory_nash, X0, minimizer_kwargs=kwargs)
        x = result.x
        avgP_wardrop, minimization = theory_nashP(A,path_list,edge_list,x)
        print(start, end, F_value(avgP_opt), F_value(avgP_wardrop))

        with open('./8node_example_WE_global_basinhopping.csv', 'a') as csvfile:
            csv.writer(csvfile).writerow([float(F0),(start,end),float(F_value(avgP_opt)),\
                                          float(F_value(avgP_wardrop))])