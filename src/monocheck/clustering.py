import numpy as np
from typing import List
from pathlib import Path 

from sklearn.cluster import KMeans


def cluster(features: np.ndarray, no_of_clusters:int= 2):
    kmeans = KMeans(n_clusters= no_of_clusters, random_state=22)
    kmeans.fit(features)
    return kmeans.labels_

def group_clusters(files_paths: List[Path], cluster_labels:np.ndarray):
    groups = {}
    for file, cluster in zip(files_paths, cluster_labels):
        file = str(file)
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups