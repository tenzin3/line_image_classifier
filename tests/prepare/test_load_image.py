import numpy as np
from pathlib import Path 
from monocheck.prepare import load_image


def test_load_image():
    DATA_DIR = Path(__file__).parent / "data"
    image_path = DATA_DIR / "image.jpg"
    img_array = load_image(image_path)
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape == (1, 224, 224, 3)


test_load_image()