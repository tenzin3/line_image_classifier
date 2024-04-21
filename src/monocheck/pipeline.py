from pathlib import Path 
from typing import List 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.prepare import load_image
from monocheck.feature_extraction import extract_features


def pipeline(image_paths:List[Path]):
    imgs_array = [load_image(image_path) for image_path in image_paths]
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    imgs_features = extract_features(imgs_array, model)
    return imgs_features


if __name__ == "__main__":
    image_path = Path("image.jpg")
    imgs_path = [image_path]

    imgs_features = pipeline(imgs_path)
    print(imgs_features)
    


