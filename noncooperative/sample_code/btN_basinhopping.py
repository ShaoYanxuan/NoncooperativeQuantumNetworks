import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
import math
import random
from itertools import chain
import csv
from scipy.optimize import fsolve
import matplotlib.ticker as mtick
from scipy.optimize import bisect
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from datetime import datetime

def F_value(p):
    return (3*p+1)/4
def p_value(F):
    return (4*F-1)/3

def find_all_path(graph, start, end):
    path_list = []
    for path in gt.all_paths(graph, start, end):
        path_list.append(list(path))
    return path_list

def func(x, S):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)-S
def solve_sch_co(S):
    lambda0 = fsolve(func, 1e-5, args=S)[0]
    return lambda0
    
def p_pure(n, cap):
    if cap == 0:
        return 0
    else:
        if n<=cap:
            return 1
        else:
            S = cap/n
            lambda0 = solve_sch_co(S)
            return (np.sqrt(lambda0*(1-lambda0))*4+1)/3

def p_mixed(p, n, cap):   # gives the p value of mixed state 
    if cap==0:
        return 0
    else:
        if n>=cap:
            return p*cap/n
        else:
            floor = int(n/cap*1000)
            return avg_mixed_p[floor]+(avg_mixed_p[floor+1]-avg_mixed_p[floor])*(n/cap-floor/1000)

def path_p(path_count_list, path_list, graph, M,edge_cap):
    #     count use of edge 
    edge_count = {}
    for edge in graph.get_edges():
        edge_count[(min(edge[0], edge[1]), max(edge[0], edge[1]))] = 0
    for i in range(len(path_list)):
        path = path_list[i]
        for j in range(len(path)-1):
            edge_count[(min(path[j],path[j+1]), max(path[j],path[j+1]))] += path_count_list[i]
    #     calculate the p value of each edge
    edge_p = {}
    for item in edge_count:
        if ent_dict[item] == 0:                # if pure states
            edge_p[item] = p_pure(edge_count[item], edge_cap[item])
        else:                                  # if mixed states
            edge_p[item] = p_mixed(p,edge_count[item],edge_cap[item])
    # calculate the fidelity of each path
    path_p_list = []
    for j in range(len(path_list)):
        path = path_list[j]
        pp = [edge_p[(min(path[i],path[i+1]),max(path[i],path[i+1]))] for i in range(len(path)-1)]
        pp = np.prod(pp)
        path_p_list.append(pp)
    return path_p_list

def Avg_p(path_p_list, path_count_list, m, M):
    return sum([path_count_list[i]*path_p_list[i] for i in range(len(path_count_list))])/(m*M)

def single_path_p(path_count_list, path_list, single_path, graph, M, edge_cap):
    #     count use of edge 
    edge_count = {}
    for edge in graph.get_edges():
        edge_count[(min(edge[0], edge[1]), max(edge[0], edge[1]))] = 0
    for i in range(len(path_list)):
        path = path_list[i]
        for j in range(len(path)-1):
            edge_count[(min(path[j],path[j+1]), max(path[j],path[j+1]))] += path_count_list[i]
    p_single = 1
    path = path_list[single_path]
    for i in range(len(path)-1):
        item = (min(path[i],path[i+1]), max(path[i],path[i+1]))
            
        if ent_dict[item] == 0:
            p_single = p_single*p_pure(edge_count[item], edge_cap[item])
        else:
            p_single = p_single*p_mixed(p,edge_count[item], edge_cap[item])
    return p_single

def order(arbitraryList):
    return arbitraryList[1]

def single_edge_p(path_list, path_count_list, graph, M, edge_cap):
    edge_p_dict = {}
    for edge in graph.get_edges():
        edge_p_dict[(min(edge[0],edge[1]),max(edge[0],edge[1]))] = 0
    for i in range(len(path_list)):
        path = path_list[i]
        for j in range(len(path)-1):
            edge_p_dict[(min(path[j],path[j+1]),max(path[j],path[j+1]))] +=path_count_list[i]
    for item in edge_p_dict:
        if ent_dict[item] == 0:
            edge_p_dict[item] = p_pure(edge_p_dict[item], edge_cap[item])
        else:
            edge_p_dict[item] = p_mixed(p,edge_p_dict[item], edge_cap[item])
    return edge_p_dict

def nash_equilibrium(path_list, path_count_list, m, graph, M, edge_cap):
    previous_path_count_list = [0 for i in range(len(path_list))]
    while previous_path_count_list != path_count_list:
        previous_path_count_list = path_count_list.copy()
        path_p_list = path_p(path_count_list, path_list, graph, M, edge_cap)
        path_p_list_low = [path_p_list[i] if path_count_list[i]>0 else 1 for i in range(len(path_count_list))]
        lowest_p_index = np.argmin(path_p_list_low); lowest_p = min(path_p_list_low)
        changing_p = []; changing_path = []
        if max(path_p_list)<lowest_p+1e-5:
            break
        for j in chain(range(lowest_p_index), range(lowest_p_index+1,len(path_p_list))):
            if path_p_list[j]>lowest_p+1e-5:
                new_path = path_count_list.copy()
                new_path[lowest_p_index] -= 1
                new_path[j] += 1
                changing_p.append(single_path_p(new_path, path_list, j, graph, M, edge_cap)); changing_path.append(j)
        if len(changing_p)>0:
            if max(changing_p)>lowest_p+1e-5:
                path_count_list[lowest_p_index] -= 1; path_count_list[changing_path[np.argmax(changing_p)]] += 1
        # print(path_count_list)
    path_p_list = path_p(path_count_list, path_list, graph, M, edge_cap)
    avg_p = Avg_p(path_p_list, path_count_list, m, M)
    return path_p_list, avg_p, path_count_list

def real_nash(start, end, m, graph, M,edge_cap):
    path_list0 = []
    for path in gt.all_paths(graph, start, end):
        path = list(path)
        edge_p = [(1-ent_dict[(min(path[i],path[i+1]),max(path[i],path[i+1]))])*(1-p_mixed(p,1,M))+p_mixed(p,1,M) for i in range(len(path)-1)]
        path_pp = np.prod(edge_p)
        path_list0.append([path, path_pp])
    path_list0.sort(reverse=True, key=order)
    path_list = [item[0] for item in path_list0 if item[1]==path_list0[0][1]]
    path_list0 = path_list0[len(path_list):]
    path_count_list = [int(m*M/len(path_list)) for i in path_list]
    path_count_list[-1] = m*M-sum(path_count_list[:-1])
    path_p_list, avg_p, path_count_list = nash_equilibrium(path_list, path_count_list, m, graph, M, edge_cap)
    previous_avg_p = 0

    if len(path_list0)>0:
        edge_p_dict = single_edge_p(path_list, path_count_list, graph, M, edge_cap)
        for i in range(len(path_list0)):
            path = path_list0[i][0]
            edge_p = [edge_p_dict[(min(path[j],path[j+1]),max(path[j],path[j+1]))] for j in range(len(path)-1)]
            path_list0[i][1] = np.prod(edge_p)
        path_list0.sort(reverse=True, key=order)
    else:
        path_list0 = [[[],0]]
    iter_count = 0
    while path_list0[0][1]>avg_p and len(path_list0)>1:
        new_path = path_list0[:min(10*iter_count+10, len([0 for item in path_list0 if item[1]>avg_p]))]
        new_path = [item[0] for item in new_path]
        path_list0 = path_list0[len(new_path):]
        previous_avg_p = avg_p
        path_count_list.extend([0 for i in new_path])
        path_list.extend(new_path)
        path_p_list, avg_p, path_count_list = nash_equilibrium(path_list, path_count_list, m, graph, M, edge_cap)

        if len(path_list0)>0:
            edge_p_dict = single_edge_p(path_list, path_count_list, graph, M, edge_cap)
            for i in range(len(path_list0)):
                path = path_list0[i][0]
                edge_p = [edge_p_dict[(min(path[j],path[j+1]),max(path[j],path[j+1]))] for j in range(len(path)-1)]
                path_list0[i][1] = np.prod(edge_p)
            path_list0.sort(reverse=True, key=order)
        else:
            path_list0 = [[[], 0]]
        iter_count += 1
    return path_list, path_p_list, avg_p, path_count_list

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

def f1(x,cap):
    if cap==0:
        return 0
    else:
        x = x/cap
        if x<=1:
            return 1
        else:
            lambda0 = solve_sch_co(1/x)
            return (np.sqrt(lambda0*(1-lambda0))*4+1)/3

def f2(x,cap):
    if cap==0:
        return 0
    else:
        if x<1:
            p_index = int(x*M)
            return avg_mixed_p[p_index] + (avg_mixed_p[p_index+1]-avg_mixed_p[p_index])*(x*M-p_index)
        else:
            return p/x

def con(x):
    return sum(x)-m
cons = {'type':'eq', 'fun': con}

def btW_func(x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    path_p_list = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        path_p = 1
        for item in edge_index:
            if edge_type[item] == 0:
                path_p = path_p*f1(edge_num[item], edge_cap[item])
            else:
                path_p = path_p*f2(edge_num[item], edge_cap[item])
        path_p_list.append(path_p)
    avgP = sum([path_p_list[i]*x[i] for i in range(len(path_p_list))])/m
    equation = 100*sum([x[i]*(threshold_p-path_p_list[i]) for i in range(len(path_p_list)) if path_p_list[i]<threshold_p])-avgP
    return equation

def btW_p(A, path_list, edge_list, x):
    edge_num = [sum([A[i][j]*x[j] for j in range(len(path_list))]) for i in range(len(A))]
    # print(edge_num)
    path_p_list = []
    for i in range(len(path_list)):
        edge_index = np.nonzero([A[j][i] for j in range(len(edge_list))])[0]
        # print(edge_index)
        path_p = 1
        for item in edge_index:
            if edge_type[item] == 0:
                path_p = path_p*f1(edge_num[item], edge_cap[item])
            else:
                path_p = path_p*f2(edge_num[item], edge_cap[item])
        path_p_list.append(path_p)
    avgP = sum([path_p_list[i]*x[i] for i in range(len(path_p_list))])/m
    equation = sum([x[i]*(path_p_list[i]-threshold_p) for i in range(len(path_p_list)) if path_p_list[i]<threshold_p])
    return avgP, equation, path_p_list


F0 = 0.95; p = p_value(F0); M = 1000; N = 8
with open('../purification/purification_F_1000.csv', 'r') as csvfile:
    avg_mixed_p = [p_value(float(item)) for item in list(csv.reader(csvfile))[-1]]

g = gt.load_graph('../networks/8node_example.gt.gz')
ent = g.edge_properties["entanglement"]
ent_dict = {}
for edge in g.get_edges():
    ent_dict[(min(edge[0],edge[1]),max(edge[0],edge[1]))] = ent[g.edge(edge[0],edge[1])]

with open('./8node_example_btN.csv', 'w') as csvfile:
    csv.writer(csvfile).writerow(['F0','(s,t)','btN','minimization','x','path fidelity'])

for start in range(N):
    for end in range(start+1,N):
        m = min(g.vertex(start).out_degree(), g.vertex(end).out_degree())
        edge_cap_list = [1000 for i in range(12)]; 
        edge_cap = {}; k = 0
        for e in g.get_edges():
            edge_cap[(min(e[0],e[1]),max(e[0],e[1]))] = edge_cap_list[k]; k += 1
        path_list, path_p_list, avg_p, path_count_list=real_nash(start, end, m, g, M, edge_cap)
        threshold_p = avg_p

        path_list_full = find_all_path(g, start, end)
        X0 = []
        for item in path_list_full:
            if item in path_list:
                X0.append(path_count_list[path_list.index(item)]/1000)
            else:
                X0.append(0)
        print(datetime.now())
        edge_list = []; edge_type = []
        for edge in g.iter_edges():
            edge_list.append(edge); edge_type.append(ent_dict[(edge[0],edge[1])])
        
        path_list = find_all_path(g, start, end)
        edge_cap = [1 for i in range(12)]
        A = [[0 for i in range(len(path_list))] for j in range(len(edge_list))]
        for i in range(len(path_list)):
            path = path_list[i]
            for j in range(len(path)-1):
                edge = list(np.sort([path[j],path[j+1]]))
                if edge in edge_list:
                    A[edge_list.index(edge)][i] = 1
        lowerBounds = 0*np.ones(len(path_list))
        upperBounds = np.ones(len(path_list))*m
        boundData=Bounds(lowerBounds,upperBounds); options={'ftol':1e-9}
        kwargs = dict(bounds=boundData, constraints=cons, method='SLSQP', options=options)
        result = basinhopping(btW_func, X0, minimizer_kwargs=kwargs)
        x_btN = result.x
        avgp_btN, minimization, pathPList_btN = btW_p(A,path_list_full,edge_list,x_btN)
        pathFList_btN = [F_value(item) for item in pathPList_btN]
        print(start, end, F_value(avg_p), F_value(avgp_btN), minimization); 
        with open('./8node_example_btN.csv', 'a') as csvfile:
            csv.writer(csvfile).writerow([float(F0),(start,end),float(F_value(avgp_btN)),float(minimization),x_btN,pathFList_btN])
        print(datetime.now())


