import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv")

import random

num_ads = 10
num_rounds = 1000
ads_selected = []
rewards_1_of_ads = [0] * num_ads
rewards_0_of_ads = [0] *  num_ads
total_reward = 0

for round in range(0,num_rounds):
    picked_ad = 0
    max_random = 0
    for ad in range(0,num_ads):
        randombeta = random.betavariate(rewards_1_of_ads[ad]+1, rewards_0_of_ads[ad]+1)
        if randombeta > max_random:
            max_random = randombeta
            picked_ad = ad
    ads_selected.append(picked_ad)
    reward = dataset.values[round][picked_ad]
    if reward == 1:
        rewards_1_of_ads[picked_ad] = rewards_1_of_ads[picked_ad] + 1
    else:
        rewards_0_of_ads[picked_ad] = rewards_0_of_ads[picked_ad] + 1
    total_reward = total_reward + reward

import matplotlib.pyplot as plt

plt.hist(ads_selected)
plt.show()

shown_so_far = [ads_selected.count(i) for i in range(0,num_ads)]
print(shown_so_far)
print(total_reward)