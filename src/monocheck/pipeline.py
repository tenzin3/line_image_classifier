import numpy as np
from pathlib import Path 
from typing import List 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from monocheck.prepare import load_image
from monocheck.feature_extraction import extract_features
from monocheck.dimension_reduction import reduce_dimension

def pipeline(image_paths:List[Path]):
    imgs_array = [load_image(image_path).squeeze(0) for image_path in image_paths]
    imgs_array = np.stack(imgs_array, axis=0)
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    imgs_features = extract_features(imgs_array, model)
    imgs_features = imgs_features.reshape(-1,4096)
    reduced_imgs_features = reduce_dimension(imgs_features)
    return reduced_imgs_features

if __name__ == "__main__":
    imgs_path = [Path("image.jpg"), Path("image2.jpg")]
    imgs_feat = pipeline(imgs_path)
    print(imgs_feat)

    


