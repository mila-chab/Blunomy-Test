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
X = np.array(df['x'])
Y = df['y']
Z = df['z']

# Center and reduce the data
x_mean, y_mean, z_mean = X.mean(), Y.mean(), Z.mean()
x_dev, y_dev, z_dev = X.std(), Y.std(), Z.std()
X = (X - x_mean)/x_dev
Y = (Y - y_mean)/y_dev
Z = (Z - z_mean)/z_dev

## Display the datasets in 3D
"""plt.figure()

axes = plt.axes(projection="3d")

axes.scatter(X, Y, Z)
"""

## Differentiate the clusters

# I try to find the normal vector of the three planes, assuming that they are parallel
model = LinearRegression()
points = np.column_stack((X,Y))
model.fit(points, Z)

a, b = model.coef_
c = model.intercept_

print('equation:', a, b,c)
n = [a, -1] # normal vector of the 3 planes

## Rotating the plane
theta = -3*math.pi/4
cos_theta = math.cos(theta)
sin_theta = math.sin(theta)
Xp = np.array(cos_theta*X + sin_theta * Y)

# Clustering

kmeans = KMeans(n_clusters = 3)
labels = kmeans.fit_predict(Xp.reshape(-1,1))  # labels of each point
centers = kmeans.cluster_centers_          # coordinate of each cluster
print("labels",labels)
plt.plot(centers, c='black', marker='X', label='centres', linestyle = 'None')

# Plotting each cluster differently
Xp_0 = [Xp[i] for i in range(len(labels)) if labels[i] == 0]
Xp_1 = [Xp[i] for i in range(len(labels)) if labels[i] == 1]
Xp_2 = [Xp[i] for i in range(len(labels)) if labels[i] == 2]
plt.plot(Xp_0, color = 'r', linestyle = 'None', marker= 'x')
plt.plot(Xp_1, color = 'g', linestyle = 'None', marker= 'x')
plt.plot(Xp_2, color = 'c', linestyle = 'None', marker= 'x')

plt.show()


