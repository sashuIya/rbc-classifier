 <h1 align="center">
  <br>
  <img src="docs/img/cells_labeling_tool.png" alt="Logo" width="200">
  <br>
    Cells Labeling Tool
  <br>
</h1>

![GitHub License](https://img.shields.io/github/license/sashuIya/rbc-classifier)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/sashuIya/rbc-classifier)

## Workflow

*  A measurement tool cuts the metadata from the bottom of the image and stores the image scaling;
*  Identify objects by running [SegmentAnything](https://github.com/facebookresearch/segment-anything) with the customizable grid layout;
*  Use HITL technique to train and improve the classifier:
    - label several objects
    - train a classifier
    - fix the results manually
    - train a classifier again

<p align="center">
    <img width="49%" src="docs/img/sample_measurement.png" alt="measurement"/>
&nbsp;
    <img width="49%" src="docs/img/sample_grid.png" alt="grid"/>
</p>

<p align="center">
    <img width="49%" src="docs/img/sample_masks.png" alt="masks"/>
&nbsp;
    <img width="49%" src="docs/img/sample_result.png" alt="result"/>
</p>

## Features

### Output stats

| **Label**      | **Count** |
| -------------- | --------- |
| red blood cell | 133       |
| echinocyte     | 7         |
| wrong          | 23        |

### Labels

* Customize a set of labels and train classifiers for any project;
* Fix the missclassification manually (either by clicking on the masks preview or by reviewing masks one by one):

<img src="docs/img/sample_fix_labels.png" alt="fixing labels" width="600"/>
