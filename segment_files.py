import pandas as pd
import cv2
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

TRAIN_DIR = 'dataset/1000'
TRAIN_FILE_NAMES = [
    '10uM_biotin_1000x__024.tif',
    '10uM_biotin_1000x__025.tif',
    '10uM_biotin_1000x__026.tif',
    '20uM_biotin_1000x__041.tif',
    '20uM_biotin_1000x__042.tif',
    '20uM_biotin_1000x__043.tif',
    '20uM_biotin_1000x__045.tif',
    '3-1k1.tif',
    '5uM_biotin_1000x__004.tif',
    '5uM_biotin_1000x__006.tif',
    '5uM_biotin_1000x__007.tif',
    '5uM_biotin_1000x__008.tif',
    'naive RBCs_1000x__060.tif',
    'naive RBCs_1000x__061.tif',
    'naive RBCs_1000x__062.tif',
]

TEST_DIR = 'dataset/1000/test'
TEST_FILE_NAMES = [
    '10uM_biotin_1000x__027.tif',
    '20uM_biotin_1000x__046.tif',
    '5uM_biotin_1000x__009.tif',
    'naive RBCs_1000x__063.tif',
]

DEVICE = 'cuda'

SAM_CHECKPOINT = 'model/sam/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'

# To cut everything below this X. Default is 2048
X_THRESHOLD_FOR_FILE = {'3-1k1.tif': 1024}

def threshold_for_file(filepath):
    filename = os.path.basename(filepath)
    if filename in X_THRESHOLD_FOR_FILE: return X_THRESHOLD_FOR_FILE[filename]

    return 2048


def get_train_test_filepaths():
    filepaths = []
    for train_file_name in TRAIN_FILE_NAMES:
        filepaths.append(os.path.join(TRAIN_DIR, train_file_name))
    for test_file_name in TEST_FILE_NAMES:
        filepaths.append(os.path.join(TEST_DIR, test_file_name))
    
    return filepaths


def segment_anything(filepath, mask_generator):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[:threshold_for_file(filepath), :]
    masks = mask_generator.generate(image)
    print('found {} masks'.format(len(masks)))

    for mask in masks:
        x, y, w, h = mask['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        mask['segmentation'] = mask['segmentation'][y : y + h + 1, x : x + w + 1]

    results_filepath = os.path.splitext(filepath)[0] + '.pkl'
    with open(results_filepath,'wb') as f:
        pickle.dump(masks, f)


def main():
    filepaths = get_train_test_filepaths()

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,
        points_per_batch=64,
        # pred_iou_thresh=0.95,
        # stability_score_thresh=0.92,
        # crop_n_layers=0,
        # crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    for index, filepath in enumerate(filepaths):
        print('segmenting {} ({} / {})'.format(filepath, index + 1, len(filepaths)))
        segment_anything(filepath, mask_generator)

    print('Done')


if __name__ == '__main__':
    main()