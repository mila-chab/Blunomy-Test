# Catenary
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

def catenary(x, x0, z0, c):
    return z0 + c * (np.cosh((x-x0)/c) - 1)

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

## Differentiate the clusters

## Rotating the plane
theta = -3*math.pi/4    # arbitrarily chosen
cos_theta = math.cos(theta)
sin_theta = math.sin(theta)
Xp = np.array(cos_theta*X + sin_theta * Y)

### Step 1: Clustering
kmeans = KMeans(n_clusters = 3)
labels = kmeans.fit_predict(Xp.reshape(-1,1))  # labels of each point
centers = kmeans.cluster_centers_          # coordinate of each cluster

#Separating the points according to their cluster
data0 = df[labels == 0]
data1 = df[labels == 1]
data2 = df[labels == 2]

## Display the datasets in 3D
plt.figure()
axes = plt.axes(projection="3d")

### Step 2: Defining the planes

for j in range(3):
# Finding the best fitting plane => making a linear regression
    data = df[labels == j]
    model = LinearRegression()
    points = np.column_stack((data['x'], data['y']))
    model.fit(points, data['z'])

    a, b = model.coef_
    c = model.intercept_

    print(f'equation of the plane: {a}x + {b}y - z + {c} = 0')
    n = [a, b, -1] # normal vector of the plane

### Step 2: Find the best fitting curve
#Let's work in the right plane
    X,Y, Z = data['x'], data['y'], data['z']

    r = (a**2 + b**2)**0.5
    sin_theta = a/r
    cos_theta = b/r
    Xp = np.array(cos_theta*X + sin_theta * Y)

#Fit the best curve
    z0_init = data['z'].min()
    x0_init = data[data['z'] == z0_init]['x'].iloc[0]
    print(x0_init)

    y0_init = data[data['z'] == z0_init]['y'].iloc[0]
    print(y0_init)
    x0_init = cos_theta*x0_init + sin_theta * y0_init
    init = [x0_init, y0_init, z0_init]
    print("initial parameters", init)

    popt, pcov = curve_fit(catenary, Xp, Z, p0 = [x0_init, z0_init, 1])
    x0, z0, c = popt

    #print(f"Best parameters : c = {c}, x0 = {x0}, z0 = {z0}")
    """plt.scatter(Xp, Z, label='Données observées', color='red')
    plt.plot(Xp, catenary(Xp, *popt), label='Modèle ajusté', color='blue', linestyle='None', marker = '+')
    plt.legend()
    plt.show()
    """
    # Inverting the rotation
    x0p = cos_theta*x0 - sin_theta * z0
    popt = x0p, z0, c
    print(f"For curve {j}, best parameters : c = {c}, x0 = {x0p}, z0 = {z0}")

    axes.scatter(data['x'], data['y'], data['z'], label='Dataset {j}')
    #plt.scatter(X, Y, Z, label='Dataset {j}')
    plt.plot(X, Y, catenary(X, *popt), label='Catenary model', linestyle='None', marker = '+')

plt.legend()
plt.show()

