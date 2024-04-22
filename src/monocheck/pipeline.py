import numpy as np
import json 

from pathlib import Path 
from typing import List 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.prepare import load_image
from monocheck.feature_extraction import extract_features
from monocheck.dimension_reduction import reduce_dimension
from monocheck.clustering import cluster, group_clusters

def pipeline(image_paths:List[Path], output_file_path:Path=Path('grouped_clusters.json')):
    imgs_array = [load_image(image_path).squeeze(0) for image_path in image_paths]
    imgs_array = np.stack(imgs_array, axis=0)
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    
    imgs_features = extract_features(imgs_array, model)
    imgs_features = imgs_features.reshape(-1,4096)
    reduced_imgs_features = reduce_dimension(imgs_features)
    """ cluster the image features with kmeans """
    clustering_labels = cluster(reduced_imgs_features)
    """ group images based on labels, with key: label and values: image paths"""
    cluster_groups = group_clusters(image_paths, clustering_labels)

    """ save the result """
    with open(output_file_path, 'w') as file:
        json.dump(cluster_groups, file, indent=4)
    return cluster_groups



if __name__ == "__main__":
    imgs_path = [Path("image.jpg"), Path("image2.jpg")]
    grouped_clusters = pipeline(imgs_path)


    


