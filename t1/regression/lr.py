import numpy as np
import pandas as pd

def retExpLR(x, y):
    sumX = x.sum()
    sumY = y.sum()
    sumXY = (x*y).sum()
    sumX2 = (x*x).sum()
    n = len(x)

    b0 = (sumY*sumX2 - sumX*sumXY)/(n*sumX2 - sumX**2)
    b1 = (n*sumXY - sumX*sumY)/(n*sumX2 - sumX**2)

    return (b0, b1)

def retExpMLR(x,y):
    xP = np.transpose(x)
    a = np.linalg.inv(np.matmul(xP,x))
    b = np.matmul(xP,y)
    return np.matmul(a,b)