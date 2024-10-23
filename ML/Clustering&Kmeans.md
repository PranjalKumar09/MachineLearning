# Clustering

## Segmentation
- **Clustering**: Grouping similar objects within a cluster, dissimilar to objects in other clusters.

### Applications
- **Retail/Marketing**: Identify customer buying patterns, recommend new products.
- **Fraud Detection**: Detect fraud in credit card usage.
- **Loyalty Analysis**: Identify loyal customers.
- **Insurance**: Assess risk of customers.
- **News Categorization**: Auto-categorize news and recommend similar articles.
- **Healthcare**: Characterize patient behavior.
- **Genetics**: Cluster genetic markers to identify family ties.
- **Exploratory Data Analysis**: Summarize data insights.

## Types of Clustering
1. **Partitioned-Based Clustering**
   - Efficient, divides data into partitions.
   - Examples: K-means, K-median, Fuzzy C-means.
   
2. **Hierarchical Clustering**
   - Produces cluster trees.
   - Examples: Agglomerative, Divisive.

3. **Density-Based Clustering**
   - Forms arbitrary-shaped clusters.
   - Example: DBSCAN.

## K-Means Clustering
- A partitioning method that divides data into non-overlapping subsets without internal structure.
- Minimizes intra-cluster distances, maximizes inter-cluster distances.
- **Process**:
  1. Randomly place K centroids (one per cluster).
  2. Calculate distance of each point from the centroids.
  3. Assign points to nearest centroid.
  4. Recalculate centroid positions.
  5. Repeat until centroids stop moving.

### Evaluating K-Means
- **External Approach**: Compare clusters with ground truth if available.
- **Internal Approach**: Measure average distance between points within clusters.

- **Note**: Number of clusters (K) must be predefined.



### Code
#### Selecting Features:
``` py
X = customer_data.iloc[:, [3, 4]].values
# print(X)
```
#### WCSS Calculation:
   - **WCSS (Within-Cluster Sum of Squares)**: This measures the compactness of the clusters. Lower values indicate more compact clusters
   - The code loops over a range of cluster numbers (from 1 to 10), fits the K-means model, and appends the WCSS (or inertia) for each number of clusters to the wcss list.
``` py
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```
#### Elbow Method Visualization:
   - This plots WCSS against the number of clusters to visualize the Elbow Point. The elbow point is where the rate of decrease sharply changes, indicating the optimal number of clusters.

``` py
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
#### Training the K-means Model:
``` py
# optimum number of clusters = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
# return a label for each data point based on their cluster 
Y = kmeans.fit_predict(X)
```
#### Visualizing the Clusters::
``` py
plt.figure(figsize=(8, 8))
plt.scatter(X[Y==0, 0], X[Y==0, 1], s=50, c='green', label='Cluster 1') 
plt.scatter(X[Y==1, 0], X[Y==1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2, 0], X[Y==2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3, 0], X[Y==3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4, 0], X[Y==4, 1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```



