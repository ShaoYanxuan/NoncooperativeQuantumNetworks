import numpy as np
import matplotlib.pyplot as plt
import csv

def purify_2_1(p1, p2):
    prob = (1+p1*p2)/2
    new_p = (p1+p2+4*p1*p2)/(3+3*p1*p2)
    return new_p, prob
def F_value(p):
    return (3*p+1)/4
def p_value(F):
    return (4*F-1)/3
    
def purification(F0, n, m):
    p_list = [p_value(F0) for i in range(n)]
    while len(p_list)>m+1:
        p_list.sort(reverse=True)
        p1 = p_list[-1]; p_list.pop(-1)
        p2 = p_list[-1]; p_list.pop(-1)
        new_p, prob = purify_2_1(p1,p2)
        if np.random.random()<prob:
            p_list.append(new_p)
    return [F_value(item) for item in p_list]
        
N = 1000
for F0 in np.arange(0.55,0.99,0.025): 
    purified_F = [1]
    for m in range(1,1001):
        purified_F.append(np.mean([np.mean(purification(F0,N,m)) for i in range(1000)]))

    with open('./purification_F_1000.csv', 'a') as csvfile:
        csv.writer(csvfile).writerow(purified_F)