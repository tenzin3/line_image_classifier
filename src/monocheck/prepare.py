import numpy as np 

from pathlib import Path 

from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 


def load_image(image_path:Path) -> np.ndarray:
    """ load the image as a 224x224 array """
    img = load_img(image_path, target_size=(224,224))
    """ convert from 'PIL.Image.Image' to numpy array """
    img = np.array(img)
    """ 3 dimensions (rows, columns, channels)-> (num_of_samples, rows, columns, channels)"""
    reshaped_img = img.reshape(1,224,224,3)
    """ preprocess for vgg16"""
    input_array = preprocess_input(reshaped_img)
    return input_array



if __name__ == "__main__":
    image_path = Path("image.jpg")
    img_array = load_image(image_path)