<h1 align="center">
  <br>
  <img src="docs/img/cells_labeling_tool.png" alt="Cells Labeling Tool Logo" width="200">
  <br>
  Cells Labeling Tool
  <br>
</h1>

![GitHub License](https://img.shields.io/github/license/sashuIya/rbc-classifier)
![GitHub Commit Activity](https://img.shields.io/github/commit-activity/y/sashuIya/rbc-classifier)

# Blood Cell Image Analysis Tool

## Objective

The primary goal of this project is to develop a comprehensive tool for the analysis and classification of blood cells in microscopy images. By leveraging advanced image processing techniques and machine learning models, we aim to automate the identification and labeling of various blood cell types, thereby enhancing the efficiency and accuracy of hematological analyses. Our tool is designed to support researchers and clinicians by providing detailed insights into blood samples, facilitating early detection of abnormalities and aiding in diagnostic decision-making processes.

### Secondary Objective

Beyond its application in hematology, this tool is designed with flexibility in mind, allowing users to define their own objects, labels, and databases for a wide range of image analysis tasks. Whether it's identifying components in automotive manufacturing, sorting biological specimens, or any other application requiring precise object classification, our tool provides a robust framework for custom training and deployment. This adaptability makes it a valuable resource for professionals across various fields seeking to leverage machine learning for image-based classification challenges.

## Workflow

- The tool begins by extracting metadata from the bottom of the image using a measurement function, which also stores the image scaling factor.
- It then identifies objects within the image through the application of [SegmentAnything](https://github.com/facebookresearch/segment-anything), utilizing a customizable grid layout for segmentation.
- To enhance the classification accuracy, the tool employs a Human-in-the-Loop (HITL) approach, which involves:
    - Manually labeling several objects to establish initial training data.
    - Training a classifier model based on this labeled data.
    - Reviewing and correcting any inaccuracies in the classification results.
    - Repeating the training process with the corrected data to refine the classifier's performance.

<p align="center">
    <img width="49%" src="docs/img/sample_measurement.png" alt="Measurement Process"/>
&nbsp;
    <img width="49%" src="docs/img/sample_grid.png" alt="Grid Layout"/>
</p>

<p align="center">
    <img width="49%" src="docs/img/sample_masks.png" alt="Masks Preview"/>
&nbsp;
    <img width="49%" src="docs/img/sample_result.png" alt="Classification Result"/>
</p>

## Features

### Output Statistics

| **Label**      | **Count** |
| -------------- | --------- |
| red blood cell | 133       |
| echinocyte     | 7         |
| wrong          | 23        |

### Customizable Labels

- Users have the flexibility to customize a set of labels according to their needs and train classifiers specifically for their projects.
- The tool allows for manual correction of misclassifications either by interacting with the masks preview or by individually reviewing and adjusting the masks.

<img src="docs/img/sample_fix_labels.png" alt="Manual Label Correction" width="600"/>

### Classification Accuracy

When training a model, you can monitor its performance through both training and validation phases. The following visualizations illustrate the model's ability to accurately classify various instances across these stages.

<img src="docs/img/train_embedding_visualization.png" alt="Training Embeddings Visualization" width="600px">
<img src="docs/img/val_embedding_visualization.png" alt="Validation Embeddings Visualization" width="600px">
