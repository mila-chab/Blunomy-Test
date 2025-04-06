# Catenary
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def catenary(x0, y0, c, x):
    return y0 + c * (math.cosh((x-x0)/c) - 1)

df_easy = pd.read_parquet('LiDAR datasets/lidar_cable_points_easy.parquet') #1502 rows * 3 columns
#print(df_easy)

"""
print("hard")
df_hard = pd.read_parquet('LiDAR datasets/lidar_cable_points_hard.parquet') #601 rows * 3 columns
print(df_hard)

print("medium")
df_medium = pd.read_parquet('LiDAR datasets/lidar_cable_points_medium.parquet') #2803 rows * 3 columns
print(df_medium)

print("extrahard")
df_extrahard = pd.read_parquet('LiDAR datasets/lidar_cable_points_extrahard.parquet') #1201 rows * 3 columns
print(df_extrahard)
"""
df = df_easy
X = df['x']
Y = df['y']
Z = df['z']

## Display the datasets in 3D
plt.figure()
axes = plt.axes(projection="3d")

axes.scatter(X, Y, Z)
plt.show()

def VectorialProd(n, point):
    """ n : vector
     point : point of space
     evalues if point belongs to the plan which normal vector is n
     """
    return n[0] * point[0] + n[1] * point[1] + n[2] * point[2]

def difference(u,v):
    return u-v

## Differentiate the clusters

# I try to find the normal vector of the three planes, assuming that they are parallel

model = LinearRegression()
points = df[['x','y']]
model.fit(points,Z)

a, b = model.coef_
c = model.intercept_

print('equation:', a, b, c)
n = [a, b, -1] # normal vector of the 3 planes

