# Blunomy-Test
# Mila Chabassier

### The datasets
size of each file : 
 easy: 1502 rows x 3 columns
 hard: 601 rows x 3 columns
 medium: 2803 rows x 3 columns
 extrahard: 1201 rows x 3 columns

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

# Numerical Results
equation of the plane: -0.2164 x + -0.1172y - z + 10.495 = 0
For curve 0, best parameters : c = 59.5034, x0 = 0.003423, z0 = 9.998

equation of the plane: 0.08073831808890823x + 0.03986659559229823y - z + 10.601664761086635 = 0
For curve 1, best parameters : c = 66.123, x0 = -0.1924, z0 = 10.00216

equation of the plane: 0.24453871495495946x + 0.13598056504781159y - z + 10.256174987190299 = 0
For curve 2, best parameters : c = 57.8119877607561, x0 = 0.23604887620660522, z0 = 10.001780812246283

The reasoning and the coding structured used on the Easy dataset were reused in the following datasets, as they could also be applied.


## The Medium dataset
The issue in this dataset is that there are two different kinds of clusters: one along the vertical axis and one along the (x, y) coordinates.
I will therefore treat this dataset the same way I have with the easy dataset, by adding one more cluster. First, I have separated the data along the z-axis, which gave me 2 clusters, which I will call the 'z-clusters'. Then, I solved the problem exactly the same way than the Easy dataset in each z-cluster. We can notice 3 curves and 4 curves respectively in the z-clusters.
One of the issue I have encountered here is that the KMeans algorithm labels the clusters in a random order. Indeed, as I have given a list of the number of clusters in each z-cluster (so, 4 and 3), sometimes the algorithm inverts the order of the labelling and the clustering doesn't work...

