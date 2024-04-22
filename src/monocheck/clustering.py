import numpy as np
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path 

from sklearn.cluster import KMeans
from keras.preprocessing.image import load_img 


def cluster(features: np.ndarray, no_of_clusters:int= 2):
    kmeans = KMeans(n_clusters= no_of_clusters, random_state=22)
    kmeans.fit(features)

    print("[SUCCESS]: Clustering succesfully done")
    return kmeans.labels_

def group_clusters(files_paths: List[Path], cluster_labels:np.ndarray):
    groups = {}
    for file, cluster in zip(files_paths, cluster_labels):
        file = str(file)
        cluster = int(cluster)   # int32 -> int conversion
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups


def view_clusters(grouped_clusters, output_dir='clustering_output'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = list(grouped_clusters.keys())
    for cluster in clusters:
        plt.figure(figsize=(30,30))

        """ file names """
        files = grouped_clusters[cluster]
        """ Only allow up to 100 images to be shown at a time """
        if len(files) > 100:
            files = files[:100]

        """ Plot each image in the cluster """
        for index, file in enumerate(files):
            plt.subplot(10, 10, index + 1)
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')

        """ save the image"""
        plt.savefig(f'{output_dir}/cluster_{cluster}.png')
        plt.close()  

        print(f"Cluster {cluster} saved to {output_dir}/cluster_{cluster}.jpg")

