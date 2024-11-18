import numpy as np

from sklearn.decomposition import PCA

def reduce_dimension(images_feature: np.ndarray, components:int = 100):
    pca = PCA(n_components=2, random_state=22)
    pca.fit(images_feature)
    reduced_images_feature = pca.transform(images_feature)
    
    print("[SUCCESS]: Image features dimensions reduction successfully done.")
    return reduced_images_feature

