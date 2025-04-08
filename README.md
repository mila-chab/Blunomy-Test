# Blunomy Application Test
# Mila Chabassier

### The datasets
Each dataset is treated in a separate .py file. To see the clustering and the models fitted on all of the curves of the dataset, the code just needs to be ran.

## Starting with the Easy package
# First step : Clustering with KMeans
The problem is the algorithm doesn't define the different clusters in the right way: the points are not separated according to their plane


what we can notice is that the vertical axis for distinguishing the points is useless, and can even induce ourselves in error! I have suppressed it.
I have then applied a rotation in the plane (x,y) of an arbitrary angle chosen by looking at the data. This allowed me to study the projection of the rotated data on the x-axis.

I could then use KMeans to cluster the data along only the x-axis, which gave a conclusive result.

# Second step : Finding the right planes
After having labelled the data, I could study them separately to establish the equation of the plane, using a Linear Regression, thaks to Scikit Learn.
The equation of the plane is defined by: a x + b y - z + c = 0
with n = [a, b, -1] being the normal vector of each plane

What we can notice is, although the three planes may seem parallel when plotting the data in the first place, the normal vector of the planes are not parallel.

The reasoning and the coding structured used on the Easy dataset were reused in the following datasets, as they could also be applied.

# Numerical Results
#Curve 0
Equation of the plane: -0.2164 x + -0.1172y - z + 10.495 = 0
Best parameters of the fit: c = 57.81, x0 = 0.236, z0 = 10.0
Standard-deviation to the model: 0.162

#Curve 1
Equation of the plane: 0.0807 x + 0.0399 y - z + 10.60 = 0
Best parameters : c = 66.1, x0 = -0.192, z0 = 10.00
Standard-deviation: 0.240

#Curve 2
Equation of the plane: 0.245 x + 0.136 y - z + 10.26 = 0
Best parameters : c = 59.5, x0 = 0.00342, z0 = 9.998
Standard-deviation: 0.152

Although I have tried different initial conditions, the fit was still not perfect. Maybe, other algorithms should be used to find a better fit. You can look at the 'Parameters of Curve 1 - Easy' image to see on one curve how the model fits to the data.


## The Medium dataset
The issue in this dataset is that there are two different kinds of clusters: one along the vertical axis and one along the (x, y) coordinates.
I will therefore treat this dataset the same way I have with the easy dataset, by adding one more cluster. First, I have separated the data along the z-axis, which gave me 2 clusters, which I will call the 'z-clusters'. Then, I solved the problem exactly the same way than the Easy dataset in each z-cluster. We can notice 3 curves and 4 curves respectively in the z-clusters.
One of the issue I have encountered here is that the KMeans algorithm labels the clusters in a random order. Indeed, as I have given a list of the number of clusters in each z-cluster (so, 4 and 3), sometimes the algorithm inverts the order of the labelling and the clustering doesn't work... So, if the clustering doesn't seem to work, the code has to be executed again.

# Numerical Results
Standard deviation of the residuals: 
For the z-cluster with 4 curves: [0.140, 0.158, 0.290,  0.129]
For the z-cluster with 3 curves: [0.279, 0.279, 0.322]


## The Hard dataset
# Numerical Results
#Curve 0
Best parameters : c = 58.42, x0 = -0.00743, z0 = 9.997
Standard deviation: 0.153

#Curve 1
Best parameters : c = 59.50, x0 = -0.232, z0 = 10.0
Standard deviation: 0.182

#Curve 2
Best parameters : c = 58.75, x0 = 0.238, z0 = 9.997
Standard deviation: 0.180


## The Extrahard dataset
This dataset semt similar to the Hard dataset, so I used the coding principle. Yet, an issue occured which I could not solve is the clustering. You can see the issue on the 'Small issue on the clustering - Extrahard' image. The purple cluster went a bit out of bound on the yellow one. Yet, it didn't seem to impact that much of the modeling curve, as you can check on the 'Parameters of Curve 2 (with a clustering issue) - Extrahard' image.

# Numerical Results
#Curve 0
Best parameters : c = 61.2, x0 = 0.144, z0 = 10.0
Standard deviation: 0.150

#Curve 1
Best parameters : c = 62.3, x0 = -0.343, z0 = 10.0
Standard deviation: 0.181

#Curve 2
Best parameters : c = 61.0, x0 = -0.0286, z0 = 10.0
Standard deviation: 0.165


## Conclusion
Working on the Easy datasets helped me find the right structure to cluster, try to fit the model and analyse the data. I have applied the same way of thinking and coding on the other dataset. Yet, the models fit to the data were not so perfect, although I have tried several initial conditions by hand to find the best ones: the standard deviation for each dataset is a bit high.
I have let you with images of some fits made on each dataset.
