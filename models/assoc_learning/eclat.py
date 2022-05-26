from unittest import result
import numpy as np, pandas as pd

def inspect(data):
    lhs = [tuple(result[2][0][0])[0] for result in data]
    rhs = [tuple(result[2][0][1])[0] for result in data]
    supports = [result[1] for result in data]
    return list(zip(lhs,rhs,supports))


dataset = pd.read_csv("../../resources/Part 5 - Association Rule Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv",header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])

from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2 )

results = list(rules)
resultsAsDataframe = pd.DataFrame(inspect(results),columns=['lhs','rhs','support'])
print(resultsAsDataframe.nlargest(n=10,columns='support'))