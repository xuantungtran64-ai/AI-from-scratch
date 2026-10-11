import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''data = [
    [1.0, 1.2], [1.3, 0.8], [0.8, 1.0], [1.2, 1.3], [0.9, 0.7],
    [5.2, 5.0], [4.8, 5.3], [5.5, 4.9], [5.1, 5.4], [4.9, 4.8],
    [9.0, 1.0], [9.3, 1.2], [8.7, 0.8], [9.1, 0.9], [8.8, 1.3],
    [2.0, 8.0], [2.2, 7.7], [1.8, 8.3], [2.4, 7.9], [2.1, 8.1]
]

df = pd.DataFrame(data, columns=['x1', 'x2'])'''


data_path = os.path.join(os.path.dirname(__file__), 'city_lifestyle_dataset.csv')
df = pd.read_csv(data_path)

x = df.drop(columns=['city_name', 'country'])


X = x.to_numpy().astype(float)
minx = X.min(axis=0)   # vector shape (n_features,)
maxx = X.max(axis=0)   # vector shape (n_features,)



def distance(a,b):
    return np.sqrt(np.sum((a - b)**2, axis = 1))

def clustering(k,X,ir):
    centroids = np.random.uniform(minx, maxx, size =(k,X.shape[1]))

    for _ in range(ir):
        y = []

        for data_points in X:
            distances = distance(centroids,data_points)
            clustered_lables = np.argmin(distances)
            y.append(clustered_lables)
        y = np.array(y)
        
        cluster_indices = []

        for i in range(k):
            cluster_indices.append(np.where(y == i)[0])

        cluster_center = []

        for i, indices in enumerate(cluster_indices):

            if len(indices) == 0:
                cluster_center.append(centroids[i])
            else:
                cluster_center.append(np.mean(X[indices],axis=0))

        if  np.max(np.abs(centroids - np.array(cluster_center))) <0.0001:
            break

        else:
            centroids = np.array(cluster_center)
    return y



ir = 1000
k = 4
print(clustering(k,X,ir))



'''y, centroids = clustering(k, X, ir)

# Vẽ scatter
plt.figure(figsize=(6,6))

for i in range(k):
    points = X[y == i]
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}")

# Vẽ centroids
plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=200, marker='X', edgecolors='black', linewidths=2)

plt.title("K-means Clustering Visualization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()'''