import numpy as np
from typing import List 
from keras.applications.vgg16 import VGG16 


def extract_features(images_input: np.ndarray, model:VGG16):
    images_features = model.predict(images_input)

    print("[SUCCESS]: Image features extraction done.")
    return images_features



