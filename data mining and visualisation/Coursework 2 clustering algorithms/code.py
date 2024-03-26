#SANTHOSH ARUNAGIRI
#201586816
#importing the needed libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(30)

"""
This function implements k-means clustering, It takes three arguments k, the given dataset and the iterations.
-It chooses the a data point from the dataset and makes it the initial centroid for all clusters.
-through all iterations the datapoints are assigned to the nearest cluster.
-Then, the function updates the centroids of each cluster by taking the mean of all data points assigned to that cluster.
-The iteration stops when either the maximum number of iterations is reached or the convergence is acquired.
"""
def kmeans(k, dataset, iterations=100):

    # Step 1: Initialization
    n = dataset.shape[0]
    m = dataset.shape[1]
    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = dataset[centroid_indices, :]
    # Step 2: Assigning
    cluster_number = np.zeros(n, dtype=int)
    for iteration in range(iterations):
        for i in range(n):
            distances = np.sum((dataset[i, :] - centroids)**2, axis=1)
            cluster_number[i] = np.argmin(distances)
            
        # Step 3: Optimization
        new_centroids = np.zeros((k, m))
        for j in range(k):
            new_centroids[j, :] = np.mean(dataset[cluster_number == j, :], axis=0)
            
        # until convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, cluster_number

"""
This function implements k-means++ clustering, It takes three arguments k, the given dataset and the iterations.
-The function first initializes the centroids, with the first one chosen uniformly at random from the dataset.
-Then it assigns each data points to the centroids with respect to the mean of data points in clusters.
-The function continues to update the centroids and assignments until convergence or until the maximum number of iterations is reached. 
"""
def k_meanspp_clustering(k, dataset, iterations=100):

    
    # Step 1: Initialization
    n = dataset.shape[0]
    m = dataset.shape[1]
    centroids = np.zeros((k, m))
    
    # Choose first centroid uniformly at random from dataset
    first_centroid_index = np.random.choice(n)
    centroids[0, :] = dataset[first_centroid_index, :]
    
    # Choose the rest of the centroids using k-means++ algorithm
    for j in range(1, k):
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = np.min(np.sum((dataset[i, :] - centroids[:j, :])**2, axis=1))
            
        probabilities = distances / np.sum(distances)
        new_centroid_index = np.random.choice(n, p=probabilities)
        centroids[j, :] = dataset[new_centroid_index, :]
    
    # Step 2: Assigning
    cluster_number = np.zeros(n, dtype=int)
    for iteration in range(iterations):
        for i in range(n):
            distances = np.sum((dataset[i, :] - centroids)**2, axis=1)
            cluster_number[i] = np.argmin(distances)
            
        # Step 3: Optimization
        new_centroids = np.zeros((k, m))
        for j in range(k):
            new_centroids[j, :] = np.mean(dataset[cluster_number == j, :], axis=0)
            
        # until convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, cluster_number

"""
This function implements Bisecting k-means clustering, It takes three arguments k, the given dataset and the iterations.
-All of the data points are first assigned to a single cluster after the centroids are initialised by the function. 
-Then, using standard k-means clustering, it ietrates to choose the cluster with the highest sum of squared errors (SSE) and divides it into two clusters. 
-The cluster indices and centroids for the full set of clusters are then updated using the obtained centroids and assignments. 
-This operation is iterated by the function until the desired number of clusters is obtained. 
"""
def bisecting_k_means(k, dataset, iterations=100):

    # Step 1: Initialization
    n = dataset.shape[0]
    m = dataset.shape[1]
    cluster_indices = [list(range(n))]
    centroids = np.zeros((1, m))
    centroids[0, :] = np.mean(dataset, axis=0)
    
    # Step 2: Repeat until k clusters have been formed
    while len(cluster_indices) < k:
        # Choose the cluster with the largest SSE for splitting
        max_sse_cluster_index = 0
        max_sse = float("-inf")
        for i, indices in enumerate(cluster_indices):
            cluster = dataset[indices, :]
            sse = np.sum((cluster - centroids[i, :])**2)
            if sse > max_sse:
                max_sse_cluster_index = i
                max_sse = sse
        
        # Split the cluster using regular k-means
        cluster_indices_to_split = cluster_indices.pop(max_sse_cluster_index)
        cluster_to_split = dataset[cluster_indices_to_split, :]
        split_centroids, split_labels = kmeans(2, cluster_to_split, iterations)
        
        # Update cluster indices and centroids  

        new_cluster_indices_1 = [cluster_indices_to_split[i] for i in range(len(cluster_indices_to_split)) if split_labels[i] == 0]
        new_cluster_indices_2 = [cluster_indices_to_split[i] for i in range(len(cluster_indices_to_split)) if split_labels[i] == 1]
        cluster_indices.append(new_cluster_indices_1)
        cluster_indices.append(new_cluster_indices_2)
        centroids = np.vstack((centroids, split_centroids))
        
    # Assign each data point to the closest cluster using the final centroids
    cluster_number = np.zeros(n, dtype=int)
    for i in range(n):
        distances = np.sum((dataset[i, :] - centroids)**2, axis=1)
        cluster_number[i] = np.argmin(distances)
            
    return centroids, cluster_number

"""
Calculates the silhouette coefficient for the dataset and all three clustering function above
-It is a metric to asses the quality of the clustering
-It basically ranges from -1 to 1. as -1 being worst and 1 being the perfect
"""
def silhouette_coefficient(dataset, cluster_number):
    
    n = dataset.shape[0]
    k = np.max(cluster_number) + 1
    a = np.zeros(n)
    b = np.zeros(n)
    
    for i in range(n):
        cluster_i = cluster_number[i]
        indices_i = np.where(cluster_number == cluster_i)[0]
        a[i] = np.mean(np.sum((dataset[i, :] - dataset[indices_i, :])**2, axis=1))
        
        b_min = np.inf
        for j in range(k):
            if j != cluster_i:
                indices_j = np.where(cluster_number == j)[0]
                b_ij = np.mean(np.sum((dataset[i, :] - dataset[indices_j, :])**2, axis=1))
                b_min = min(b_min, b_ij)
        b[i] = b_min
        
    s = (b - a) / np.maximum(a, b)
    return np.mean(s)


#Load data from file
dataset_path = "dataset"
data = pd.read_csv(dataset_path, delimiter=" ", header=None)
dataset = data.iloc[:, 1:].values

#Try clustering for k=1 to k=9 and compute the Silhouette coefficient using k-means
silhouette_scores_kmeans = []
for k in range(1, 10):
    centroids, cluster_number = kmeans(k, dataset, iterations=100)
    silhouette_scores_kmeans.append(silhouette_coefficient(dataset, cluster_number))
    

#Try clustering for k= 1 to 9 and compute the Silhouette coefficient using k-means++
silhouette_scores_kmeansplusplus = []
for k in range(1, 10):
    centroids, cluster_number = k_meanspp_clustering(k, dataset, iterations=100)
    silhouette_scores_kmeansplusplus.append(silhouette_coefficient(dataset, cluster_number))


#Try clustering for k= 1 to 9 and compute the Silhouette coefficient using bisecting  k-means
silhouette_scores_bisectingkmeans = []
for k in range(1, 10):
    centroids, cluster_number = bisecting_k_means(k, dataset, iterations=100)
    silhouette_scores_bisectingkmeans.append(silhouette_coefficient(dataset, cluster_number))

# create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# plot the results for k-means in the first subplot
axs[0].plot(range(1, 10), silhouette_scores_kmeans, 'bo-', label='k-means')
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Silhouette coefficient')
axs[0].legend()

# plot the results for k-means++ in the second subplot
axs[1].plot(range(1, 10), silhouette_scores_kmeansplusplus, 'ro-', label='k-means++')
axs[1].set_xlabel('Number of clusters (k)')
axs[1].set_ylabel('Silhouette coefficient')
axs[1].legend()

# plot the results for bisecting k-means++ in the third subplot
axs[2].plot(range(1, 10), silhouette_scores_bisectingkmeans, 'mo-', label='bisecting k-means')
axs[2].set_xlabel('Number of clusters (k)')
axs[2].set_ylabel('Silhouette coefficient')
axs[2].legend()


# adjust the spacing between the subplots
plt.subplots_adjust(wspace=0.4)
# show the figure
plt.show()