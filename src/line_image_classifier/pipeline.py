import numpy as np
import json 
import pickle 
from pathlib import Path 
from typing import List 
from PIL import Image

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from line_image_classifier.prepare import load_image
from line_image_classifier.feature_extraction import extract_features
from line_image_classifier.dimension_reduction import reduce_dimension
from line_image_classifier.clustering import cluster, group_clusters, view_clusters

IMAGE_FEATURES_PICKLE = Path('array.pkl')

def classify_with_feature(image_paths:List[Path], output_file_path:Path=Path('grouped_clusters.json')):
    imgs_array = [load_image(image_path).squeeze(0) for image_path in image_paths]
    imgs_array = np.stack(imgs_array, axis=0)
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    """ load image features pickle if exists """
    if not IMAGE_FEATURES_PICKLE.exists():
        imgs_features = extract_features(imgs_array, model)
        imgs_features = imgs_features.reshape(-1,4096)
        """ save the image features as pickle file """
        with open(IMAGE_FEATURES_PICKLE, 'wb') as file:
            pickle.dump(imgs_features, file)
    else:
        with open(IMAGE_FEATURES_PICKLE, 'rb') as file:
            imgs_features = pickle.load(file)
        
    reduced_imgs_features = reduce_dimension(imgs_features)
    """ cluster the image features with kmeans """
    clustering_labels = cluster(reduced_imgs_features)
    """ group images based on labels, with key: label and values: image paths"""
    cluster_groups = group_clusters(image_paths, clustering_labels)

    """ save the result """
    with open(output_file_path, 'w') as file:
        json.dump(cluster_groups, file, indent=4)
    return cluster_groups


def get_image_size(image_path: Path):
    """Get the dimensions of an image."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def classify_with_size(image_paths: List[Path], output_file_path: Path = Path('size_based_clusters.json'), n_clusters: int = 3):
    # Extract image sizes (width and height) for clustering
    image_sizes = np.array([get_image_size(image_path) for image_path in image_paths])
    
    # Perform clustering using KMeans on image sizes
    clustering_labels = cluster(image_sizes, n_clusters)
    
    # Group images based on labels
    cluster_groups = group_clusters(image_paths, clustering_labels)
    
    # Save the result
    with open(output_file_path, 'w') as file:
        json.dump(cluster_groups, file, indent=4)
    
    return cluster_groups




