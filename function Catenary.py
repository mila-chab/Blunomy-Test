# Catenary
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def catenary(x0, y0, c, x):
    return y0 + c * (math.cosh((x-x0)/c) - 1)

df_easy = pd.read_parquet('LiDAR datasets/lidar_cable_points_easy.parquet') #1502 rows * 3 columns

df = df_easy
X = df['x']
Y = df['y']
Z = df['z']

## Display the datasets in 3D
plt.figure()
"""
axes = plt.axes(projection="3d")

axes.scatter(X, Y, Z)
"""
plt.plot(X,Y, marker='x', linestyle = 'None')

def VectorialProd(n, point):
    """ n : vector
     point : point of space
     evalues if point belongs to the plan which normal vector is n
     """
    return n[0] * point[0] + n[1] * point[1] + n[2] * point[2]

## Differentiate the clusters

# I try to find the normal vector of the three planes, assuming that they are parallel
model = LinearRegression()
points = df[['x','y']]
model.fit(points,Z)

a, b = model.coef_
c = model.intercept_

print('equation:', a, b, c)
n = [a, b, -1] # normal vector of the 3 planes

# Clustering
cluster_init = [[-12.08, 20.20],
                [-11.152, 20.39],
                [-10.39, 20.72]]

kmeans = KMeans(n_clusters = 3, init = cluster_init)
labels = kmeans.fit_predict(df[['x','y']])  # labels de chaque point
centers = kmeans.cluster_centers_
print(centers)
plt.plot(centers[0,0], centers[0,1], c='black', marker='X', label='centres')
plt.plot(centers[1,0], centers[1,1], c='black', marker='X', label='centres')
plt.plot(centers[2,0], centers[2,1], c='black', marker='X', label='centres')


plt.show()


