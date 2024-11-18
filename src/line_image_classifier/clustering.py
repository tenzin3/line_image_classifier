import numpy as np

from typing import List
from pathlib import Path 
from PIL import Image

from sklearn.cluster import KMeans


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def cluster(features: np.ndarray, no_of_clusters:int= 2):
    kmeans = KMeans(n_clusters= no_of_clusters, random_state=22)
    kmeans.fit(features)

    print("[SUCCESS]: Clustering succesfully done")
    return kmeans.labels_

def group_clusters(files_paths: List[Path], cluster_labels:np.ndarray):
    groups = {}
    for file, cluster in zip(files_paths, cluster_labels):
        file = str(file)
        cluster = int(cluster)   # int32 -> int conversion
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups

def view_clusters(grouped_clusters, save_path='output_images'):
    """ save each clustering in different pdfs """
    clusters = list(grouped_clusters.keys())
    for cluster in clusters:
        files = grouped_clusters[cluster]
        images_per_page = 10
        page_width, page_height = letter  # Default letter size
        
        """ Create a PDF canvas """
        c = canvas.Canvas(f'{save_path}/cluster_{cluster}.pdf', pagesize=letter)
        
        y_position = page_height - 72  # Initial top margin offset
        for index, file in enumerate(files):
            if index % images_per_page == 0 and index != 0:
                c.showPage()  # Add a new page if the current one is filled
                y_position = page_height - 72  # Reset position at the top of a new page
            
            """ Open image and get its original size """
            img = Image.open(file)
            img_width, img_height = img.size
            
            """ Check if the image width exceeds the page width """
            if img_width > page_width - 144:
                scale_factor = (page_width - 144) / img_width
                img_width *= scale_factor
                img_height *= scale_factor
            
            """ Draw image on the canvas at original size """
            c.drawImage(file, 72, y_position - img_height, width=img_width, height=img_height)
            
            """  Update y_position for the next image """
            y_position -= (img_height + 10)  # Move down by the image height plus some margin
        
        """save pdf"""
        c.save()  
        print(f"Cluster {cluster} images saved to {save_path}/cluster_{cluster}.pdf")
