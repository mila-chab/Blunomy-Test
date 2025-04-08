import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def catenary(x, x0, z0, c):
    return z0 + c * (np.cosh((x-x0)/c) - 1)

df = pd.read_parquet('LiDAR datasets/lidar_cable_points_extrahard.parquet')

X, Y, Z = df['x'], df['y'], df['z']

plt.figure()
axes = plt.axes(projection="3d")
axes.scatter(df['x'], df['y'], df['z'])
plt.legend()
plt.show()

