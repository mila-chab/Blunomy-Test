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
plt.figure()

axes = plt.axes(projection="3d")

axes.scatter(X, Y, Z)

def VectorialProd(n, point):
    """ n : vector
     point : point of space
     evalues if point belongs to the plan which normal vector is n
     """
    return n[0] * point[0] + n[1] * point[1] + n[2] * point[2]

## Differentiate the clusters

# I try to find the normal vector of the three planes, assuming that they are parallel
model = LinearRegression()
points = np.column_stack((X,Y))
model.fit(points, Z)

a, b = model.coef_
c = model.intercept_

print('equation:', a, b,c)
n = [a, -1] # normal vector of the 3 planes

# Clustering
cluster_init = [[0.1299, -0.4865, -0.0819],
                [-0.0075, -0.1542, -0.364],
                [0.0616, -0.1116, -0.3255]]

kmeans = KMeans(n_clusters = 3, init = cluster_init)
labels = kmeans.fit_predict(np.column_stack((X,Y,Z)))  # labels de chaque point
centers = kmeans.cluster_centers_
print(centers)
plt.plot(centers[0,0], centers[0,1], centers[0,2], c='black', marker='X', label='centres')
plt.plot(centers[1,0], centers[1,1], centers[1,2], c='black', marker='X', label='centres')
plt.plot(centers[2,0], centers[2,1], centers[2,2],c='black', marker='X', label='centres')


plt.show()


