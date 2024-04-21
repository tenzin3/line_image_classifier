import numpy as np
from typing import List 
from pathlib import Path 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.prepare import load_image

def extract_features(images_input: List[np.ndarray]):
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    images_features = []
    for image_input in images_input:
        feature = model.predict(image_input)
        images_features.append(feature)

    return images_features


if __name__ == "__main__":
    image_path = Path("image.jpg")
    img_array = [load_image(image_path)]

    features = extract_features(img_array)
    print(features)

