from cmath import log
import math
import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv")
dataset = dataset.values

def del_i(n,N):
    return math.sqrt((3*math.log(n))/(2*N))

N_users = 10000
N_ads = 10
rewards_so_far = dataset[0,:]
N_so_far = list(N_ads*[1])
UCB = list([0+del_i(i,1)] for i in range(1,N_ads+1))
current_round = N_ads+1
max_rounds_allowed = 10000

for current_round in range(N_ads,max_rounds_allowed+1):
    current_ad = UCB.index(max(UCB))
    N_so_far[current_ad] = N_so_far[current_ad]+1
    rewards_so_far[current_ad] = rewards_so_far[current_ad] + dataset[N_so_far[current_ad]][current_ad]
    r_avg = rewards_so_far[current_ad] / N_so_far[current_ad]
    UCB[current_ad] = r_avg + del_i(current_round, N_so_far[current_ad])
    
print(UCB)
print(UCB.index(max(UCB)))
print(N_so_far)