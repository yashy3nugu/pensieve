from ast import operator
import os

path = './test_results_gs/1/log_sim_rl_bus.ljansbakken-oslo-report.2010-09-29_1622CEST.log_180'
reward = []

    
f = open(path,'rb')
for line in f:
    parse = line.split()
    try:
        reward.append(float(parse[-1]))
    except IndexError:
        break
    

