import numpy as np
from sklearn.cluster import KMeans

def cluster(features: np.ndarray, no_of_clusters:int= 2):
    kmeans = KMeans(n_clusters= no_of_clusters, random_state=22)
    kmeans.fit(features)
    return kmeans.labels_