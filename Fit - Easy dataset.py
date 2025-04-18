import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def catenary(x, x0, z0, c):
    return z0 + c * (np.cosh((x-x0)/c) - 1)

df_easy = pd.read_parquet('LiDAR datasets/lidar_cable_points_easy.parquet')

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

## Display the datasets in 3D
plt.figure()
axes = plt.axes(projection="3d")

variance = np.zeros(3)
### Step 2: Defining the planes
for j in range(3):
# Finding the best fitting plane => making a linear regression
    data = df[labels == j]
    model = LinearRegression()
    points = np.column_stack((data['x'], data['y']))
    model.fit(points, data['z'])

    a, b = model.coef_
    c = model.intercept_

### Step 3: Find the best fitting curve
#Rotating the coordinates to work in the right plane
    X, Y, Z = data['x'], data['y'], data['z']

    r = (a**2 + b**2)**0.5
    sin_theta = a/r
    cos_theta = b/r
    Xp = np.array(cos_theta*X + sin_theta * Y) # rotated coordinates on the x-axis

#Fit the best curve
    z0_init = data['z'].min()
    x0_init = data[data['z'] == z0_init]['x'].iloc[0]
    y0_init = data[data['z'] == z0_init]['y'].iloc[0]

    x0_rotated_init = cos_theta*x0_init + sin_theta * y0_init
    
    popt, pcov = curve_fit(catenary, Xp, Z, p0 = [x0_rotated_init, z0_init, 20])
    x0, z0, c = popt

    # Inverting the rotation
    x0p = cos_theta*x0 
    print(f"For curve {j}, best parameters : c = {c}, x0 = {x0p}, z0 = {z0}")
    Zpred = catenary(X, x0p, z0, c)

    axes.scatter(data['x'], data['y'], data['z'], label=f'Dataset {j}')
    plt.plot(X, Y, Zpred, label=f'Catenary model {j}', linestyle='None', marker = '+')
    plt.plot(x0_init, y0_init, z0_init,  linestyle='None', marker = '+', color = 'grey', ms=10)     # plot of the center of the curve
    plt.plot(x0p, y0_init, z0_init,  linestyle='None', marker = '+', color = 'black', ms=10)        # plot of the center of the model

### Step 4: Studying the deviation to the model
    residuals = Z - Zpred
    variance[j] = np.sum(residuals**2)/ (len(residuals) - 3) # 3 parameters in the model

sdev = np.sqrt(variance)

print(f"Standard deviation to the model: {sdev}")

plt.legend()
plt.show()
