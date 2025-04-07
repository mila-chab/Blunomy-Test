import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def catenary(x, x0, z0, c):
    return z0 + c * (np.cosh((x-x0)/c) - 1)

df = pd.read_parquet('LiDAR datasets/lidar_cable_points_hard.parquet')

X, Y, Z = df['x'], df['y'], df['z']

plt.figure()
axes = plt.axes(projection="3d")
axes.scatter(df['x'], df['y'], df['z'])

## Rotating the plane
theta = math.pi/6 - math.pi/100   # still arbitrarily chosen
cos_theta = math.cos(theta)
sin_theta = math.sin(theta)
Xp = np.array(cos_theta*X + sin_theta * Y)
Yp = np.array(sin_theta*X - cos_theta * Y)

kmeans = KMeans(n_clusters = 3)
labels = kmeans.fit_predict(Xp.reshape(-1,1))  # labels of each point

## Display the datasets in 3D
axes.scatter(df['x'], df['y'], df['z'], c=labels)
axes.scatter(Xp, Yp, c = 'g', label='rotated data')
plt.legend()
plt.show()


