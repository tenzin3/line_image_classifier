import numpy as np
from pathlib import Path 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.feature_extraction import extract_features

def generate_random_image():
    """ Generate a random array with shape (1, 224, 224, 3) """
    random_image = np.random.rand(1, 224, 224, 3)
    return random_image

def test_extract_features():
    """5 random images array"""
    imgs_array = [generate_random_image() for _ in range(5)]
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    
    images_feature = extract_features(imgs_array, model)

    for image_feature in images_feature:     
        assert isinstance(image_feature, np.ndarray)
        assert image_feature.shape == (1, 4096)

