
<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

<!-- Replace with 1-sentence description about what this tool is or does.-->

<h3 align="center">monocheck</h3>


## Reference
Codes are refered from https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
## Description

Clustering is a machine learning technique used to filter out incorrectly cropped line images in applications like document scanning or OCR systems. By grouping images based on features like text alignment and white space, clustering algorithms can identify clusters of likely erroneous crops. This allows for automated detection of improperly cropped images, enhancing the efficiency and accuracy of digital document processing without needing labeled training data.

### Pecha image example:
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

### So, now what ?
The clustering analyzes these segmented line images by examining features that differentiate well-cropped lines from poorly cropped ones. Features may include the consistency of text alignment, the uniformity of text height, and the absence of cut-off text. The clustering algorithm groups similar line images together based on these features.

The primary benefit of this clustering tool is its ability to automatically identify and filter out the "bad" clusters—those groups that likely contain incorrectly cropped images. By examining the characteristics of these clusters, the system can flag these as erroneous without manual intervention. This filtering significantly improves the quality of the data input into further processing stages, such as text recognition in OCR systems, by ensuring only correctly cropped lines are used, thereby enhancing both accuracy and efficiency in document digitization workflows.









