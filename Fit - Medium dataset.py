import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def catenary(x, x0, z0, c):
    return z0 + c * (np.cosh((x-x0)/c) - 1)

df = pd.read_parquet('LiDAR datasets/lidar_cable_points_medium.parquet')

X, Y, Z = df['x'], df['y'], df['z']
# Center and reduce the data
x_mean, y_mean, z_mean = X.mean(), Y.mean(), Z.mean()
x_dev, y_dev, z_dev = X.std(), Y.std(), Z.std()
X = (X - x_mean)/x_dev
Y = (Y - y_mean)/y_dev
Z = (Z - z_mean)/z_dev


### Step 1.a: Clustering along the z-axis
kmeans = KMeans(n_clusters = 2)
labels_z = kmeans.fit_predict(np.array(Z).reshape(-1,1))  # labels of each point

### Step 1.b: Clustering on both clusters
list_n_clusters = [4, 3]

for i in range(2):
    datai = df[labels_z == i]
    X0, Y0 = datai['x'], datai['y']

    ## Rotating the plane
    theta = math.pi/6 - math.pi/100    # still arbitrarily chosen
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    Xp = np.array(cos_theta*X0 + sin_theta * Y0)

    Yp = np.array(sin_theta*X0 - cos_theta * Y0)

    kmeans = KMeans(n_clusters = list_n_clusters[i])
    labelsi = kmeans.fit_predict(Xp.reshape(-1,1))  # labels of each point
    centers = kmeans.cluster_centers_


    ## Display the datasets in 3D
    plt.figure()
    axes = plt.axes(projection="3d")

    ### Step 2: Defining the planes

    for j in range(list_n_clusters[i]):
    # Finding the best fitting plane => making a linear regression
        data = datai[labelsi == j]
        model = LinearRegression()
        points = np.column_stack((data['x'], data['y']))
        model.fit(points, data['z'])

        a, b = model.coef_
        c = model.intercept_

        print(f'equation of the plane: {a}x + {b}y - z + {c} = 0')

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
        y0_init = data[data['z'] == z0_init]['y'].iloc[0]
        x0_init = cos_theta*x0_init + sin_theta * y0_init
        init = [x0_init, y0_init, z0_init]

        popt, pcov = curve_fit(catenary, Xp, Z, p0 = [x0_init, z0_init, 1])
        x0, z0, c = popt
        
        # Inverting the rotation
        x0p = cos_theta*x0 - sin_theta * z0
        popt = x0p, z0, c
        print(f"For curve {j}, best parameters : c = {c}, x0 = {x0p}, z0 = {z0}")

        axes.scatter(data['x'], data['y'], data['z'], label=f'Dataset {j}')
        plt.plot(X, Y, catenary(X, *popt), label='Catenary model {j}', linestyle='None', marker = '+')

    plt.show()

