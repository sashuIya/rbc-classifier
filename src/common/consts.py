import os

## Features dataframe
Y_COLUMN = "y"
X_COLUMN_PREFIX = "x_"
LABELING_MODE_COLUMN = "labeling_mode"
MASK_ID_COLUMN = "mask_id"

## Labels
LABEL_UNLABELED = "unlabeled"
LABEL_WRONG = "wrong"

## Labeling modes
# labeled by human
LABELING_MANUAL = "manual"
# labeled by classification model
LABELING_AUTO = "auto"
# labeled by classification model and checked by human
LABELING_APPROVED = "semi-auto"

DATA_PATH = os.path.normpath("./dataset")
RAW_IMAGES_DIR = os.path.join(DATA_PATH, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_PATH, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_PATH, "processed")

IMAGES_METADATA_FILEPATH = os.path.join(DATA_PATH, "images_metadata.csv")
LABELS_METADATA_FILEPATH = os.path.join(DATA_PATH, "labels_metadata.csv")

EMBEDDERS_METADATA_FILEPATH = os.path.normpath("model/embedders_metadata.csv")

CLASSIFIER_CHECKPOINT_DIR = os.path.normpath("model/cells_classifier")
