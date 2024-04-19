import numpy as np 

from pathlib import Path 

from keras.preprocessing.image import load_img 


def load_image(image_path:Path) -> np.array:
    """ load the image as a 224x224 array """
    img = load_img(image_path, target_size=(224,224))
    """ convert from 'PIL.Image.Image' to numpy array """
    img = np.array(img)
    """ 3 dimensions (rows, columns, channels)-> (num_of_samples, rows, columns, channels)"""
    reshaped_img = img.reshape(1,224,224,3)
    return reshaped_img


if __name__ == "__main__":
    image_path = Path("image.jpg")
    img_array = load_image(image_path)