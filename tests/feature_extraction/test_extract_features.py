import numpy as np

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.feature_extraction import extract_features


def generate_random_image():
    """ Generate a random array with shape (1, 224, 224, 3) """
    random_image = np.random.rand(1, 224, 224, 3)
    return random_image

def test_extract_features():
    """5 random images array"""
    num_of_imgs = 5
    imgs_array = [generate_random_image().squeeze(0) for _ in range(num_of_imgs)]
    imgs_array = np.stack(imgs_array, axis=0)

    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    
    images_feature = extract_features(imgs_array, model)

    
    assert isinstance(images_feature, np.ndarray)
    assert images_feature.shape == (num_of_imgs, 4096)

