import numpy as np
from typing import List 
from keras.applications.vgg16 import VGG16 


def extract_features(images_input: np.ndarray, model:VGG16):
    """ total number of images"""
    num_images = images_input.shape[0]
    """ batch size """
    batch_size = 1000
    all_features = []

    """ predict in batches """
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)  # Ensure the last batch is handled properly
        batch_features = model.predict(images_input[start:end])
        all_features.append(batch_features)
        print(f"[{end}/{num_images}] image features extraction done.")
    
    all_features = np.vstack(all_features)
    
    print("[SUCCESS]: Image features extraction done.")
    return all_features


