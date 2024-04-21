import numpy as np
from typing import List 
from keras.applications.vgg16 import VGG16 


def extract_features(images_input: List[np.ndarray], model:VGG16):
    images_features = []
    for image_input in images_input:
        feature = model.predict(image_input)
        images_features.append(feature)

    return images_features



