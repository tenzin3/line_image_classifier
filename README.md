

<!-- Replace with 1-sentence description about what this tool is or does.-->

<h3 align="center">Line Image Classifier</h3>

![image](https://github.com/user-attachments/assets/92d6604a-2b29-4ab5-ad76-56d860eced23)

## Introduction
In the OCR domain, line segmentation models are used to divide images containing text into individual lines. However, these models are not always perfect. To address this, I developed a package to classify line images as either good or bad based on their quality.

## Description

Clustering is an unsupervised machine learning technique used to group similar data points. In this package, I applied K-means clustering to OCR. This package enables the automated detection of poorly cropped images, improving the efficiency and accuracy of digital document processing without requiring labeled training data.

### Input: Pecha image:
![I1KG812750008](https://github.com/tenzin3/monocheck/assets/52460417/0300696e-eebf-4343-a905-9a4be44bc3ae)


### Line image detection model outputs:>
![image](https://github.com/tenzin3/monocheck/assets/52460417/6506364f-3a16-41a8-8ef1-b292ebe573f1)
![image](https://github.com/tenzin3/monocheck/assets/52460417/42ba14b5-d960-4c8f-b30e-04016e5316c1)
![image](https://github.com/tenzin3/monocheck/assets/52460417/21d08777-e631-4d62-a21f-bedc9e7695b4)
![image](https://github.com/tenzin3/monocheck/assets/52460417/11b996df-c3bc-49cd-8691-beb72b8d7bea)
![image](https://github.com/tenzin3/monocheck/assets/52460417/5460abc0-2690-4d21-ad1a-5cedce893609)
![image](https://github.com/tenzin3/monocheck/assets/52460417/684798fa-9abb-401f-af61-08c13f759408)
![image](https://github.com/tenzin3/monocheck/assets/52460417/6d396ab3-3602-4b06-9021-1e71bfa6ef39)
![image](https://github.com/tenzin3/monocheck/assets/52460417/f3560fc9-f940-4e71-90d3-f6873310e0cd)

## Installation
```bash
pip install git+https://github.com/tenzin3/line_image_classifier.git
```

## Classify based on Image Size
In most cases, desired line image outputs have similar dimensions, making it efficient to cluster images based on their size. This method works well for identifying outliers with significantly different dimensions.

```python
from pathlib import Path 
from line_image_classifier.pipeline import classify_with_size

images = list(Path("ocr_images").rglob("*.json"))
output_path = Path('size_based_clusters.json')
classify_with_size(images, output_path)
```

## Classify based on the Image feature
In some cases, bad line images may have similar dimensions to good ones but are incorrect due to issues like rotation or excessive zoom. For these scenarios, classification based on image features is essential. VGG16 is used to extract image features, followed by dimensionality reduction using PCA, and clustering is then performed to group similar images effectively.

```python
from line_image_classifier.pipeline import classify_with_feature

images = list(Path("ocr_images").rglob("*.json"))
output_path = Path('feature_based_clusters.json')
classify_with_feature(images, output_path)
```


## Output
A JSON file with cluster numbers as keys and lists of image paths belonging to each group as values.
PDF files, with each PDF containing line images (10 images per page) from a specific cluster group, for better visualization of result.



