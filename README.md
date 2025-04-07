# Blunomy-Test

## The datasets
size of each file : 
 easy: 1502 rows x 3 columns
 hard: 601 rows x 3 columns
 medium: 2803 rows x 3 columns
 extrahard: 1201 rows x 3 columns

First step : CLustering with KMeans
The problem is the algorithm doesn't define the different clusters in the right way: the points are not separated according to their plane


what we can notice is that the vertical axis for distinguishing the points is useless, and can even induce ourselves in error! I have suppressed it.

## The Medium dataset
The issue here is that there are two different kinds of clusters: one vertical and one horizontal.
I will therefore treat this dataset the same way I have with the easy dataset, by adding one more cluster.
