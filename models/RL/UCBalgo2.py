from cmath import log
import math
import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv")
dataset = dataset.values

def del_i(n,N):
    return math.sqrt((3*math.log(n))/(2*N))

N_ads = 10
max_rounds_allowed = 1000
ads_selected = []
number_of_selections = [0] * N_ads
sums_of_rewards = [0] * N_ads
total_reward = 0 ## this is missing
    
for n in range(0,max_rounds_allowed):
    ad = 0
    max_upper_bound = 0
    for i in range(0, N_ads):
        if(number_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = del_i(n+1,number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

import matplotlib.pyplot as plt
plt.hist(ads_selected)
plt.show()

shown_so_far = [ads_selected.count(i) for i in range(0,N_ads)]
print(shown_so_far)
print(total_reward)