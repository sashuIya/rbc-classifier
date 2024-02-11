import glob
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import cv2
import faiss
import numpy as np
import pandas as pd
import torch

from consts import (
    LABEL_UNLABELED,
    LABELING_APPROVED,
    LABELING_MANUAL,
    LABELING_MODE_COLUMN,
    Y_COLUMN,
)

IMAGES_METADATA_FILEPATH = os.path.normpath("dataset/images_metadata.csv")
LABELS_METADATA_FILEPATH = os.path.normpath("dataset/labels_metadata.csv")
EMBEDDERS_METADATA_FILEPATH = os.path.normpath("model/embedders_metadata.csv")

CLASSIFIER_CHECKPOINT_DIR = "model/cells_classifier/"


def get_rel_filepaths_from_subfolders(folder_path, extension, exclude=None):
    search_pattern = folder_path + "/**/*.{}".format(extension)
    filepaths = glob.glob(search_pattern, recursive=True)
    if exclude is not None:
        filepaths = [filepath for filepath in filepaths if exclude not in filepath]
    return sorted(filepaths)


def get_masks_filepath(image_filepath, suffix):
    return os.path.splitext(image_filepath)[0] + suffix + ".pkl"


def get_masks_features_filepath(image_filepath, suffix=""):
    return os.path.splitext(image_filepath)[0] + suffix + "_features.csv"


def get_classifier_model_filepaths():
    filepaths = get_rel_filepaths_from_subfolders(
        folder_path=CLASSIFIER_CHECKPOINT_DIR, extension="pth"
    )
    filepaths.sort(key=os.path.getctime, reverse=True)

    return filepaths


def read_masks_features(image_filepath, suffix=""):
    masks_features_filepath = get_masks_features_filepath(image_filepath, suffix=suffix)
    if not os.path.exists(masks_features_filepath):
        return None

    return pd.read_csv(masks_features_filepath, index_col=None)


def write_masks_features(
    masks_features_df: pd.DataFrame, image_filepath: str, suffix=""
):
    return masks_features_df.to_csv(
        get_masks_features_filepath(image_filepath, suffix=suffix),
        index=False,
        header=True,
    )


def read_images_metadata():
    if not os.path.exists(IMAGES_METADATA_FILEPATH):
        return pd.DataFrame(columns=["filepath"])

    return pd.read_csv(IMAGES_METADATA_FILEPATH, index_col=None)


def write_images_metadata(images_metadata: pd.DataFrame):
    return images_metadata.to_csv(IMAGES_METADATA_FILEPATH, index=False, header=True)


def read_image(image_filepath, with_alpha=False):
    image = cv2.imread(image_filepath)
    if with_alpha:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_masks(masks_filepath):
    with open(masks_filepath, "rb") as f:
        masks = pickle.load(f)
    return masks


def read_masks_for_image(image_filepath, suffix=""):
    masks_filepath = get_masks_filepath(image_filepath, suffix)
    with open(masks_filepath, "rb") as f:
        masks = pickle.load(f)

    sorted_masks = sorted(masks, key=(lambda x: x["area"]))
    for i, mask in enumerate(sorted_masks):
        mask["id"] = i

    return sorted_masks


class EmbedderMetadata:
    _MODEL_NAME = "Model name"
    _TIME_CREATED = "Time created"
    _DATA_USED = "Data used"
    _VALIDATION_ACCURACY = "Validation accuracy"
    _EPOCHS = "Number of epochs"

    def load_embedder_metadata(self, filepath: str) -> pd.Series:
        if not os.path.exists(EMBEDDERS_METADATA_FILEPATH):
            return None

        filename = os.path.basename(filepath)
        metadata_df = pd.read_csv(EMBEDDERS_METADATA_FILEPATH, index_col=None)
        metadata = metadata_df.loc[metadata_df[self._MODEL_NAME] == filename]

        if metadata.empty:
            return None

        return metadata.iloc[0]

    def save_embedder_metadata(
        self,
        filename: str,
        timestamp: datetime,
        labeled_data_df: pd.DataFrame,
        validation_accuracy: float,
        epochs: int,
    ):
        metadata_df = pd.DataFrame(
            columns=[
                self._MODEL_NAME,
                self._TIME_CREATED,
                self._VALIDATION_ACCURACY,
                self._EPOCHS,
                self._DATA_USED,
            ]
        )
        if os.path.exists(EMBEDDERS_METADATA_FILEPATH):
            metadata_df = pd.read_csv(EMBEDDERS_METADATA_FILEPATH, index_col=None)

        # Use numpy.unique() to get unique elements and their counts
        unique_elements, counts = np.unique(
            labeled_data_df[Y_COLUMN], return_counts=True
        )
        data_used = "\t".join(
            [
                "{}: {}".format(element, count)
                for element, count in zip(unique_elements, counts)
            ]
        )

        metadata = {
            self._MODEL_NAME: filename,
            self._TIME_CREATED: timestamp.strftime("%d %b %Y (%H:%M:%S)"),
            self._VALIDATION_ACCURACY: validation_accuracy,
            self._EPOCHS: epochs,
            self._DATA_USED: data_used,
        }
        metadata_df = pd.concat(
            [metadata_df, pd.DataFrame([metadata])], ignore_index=True
        )

        metadata_df.to_csv(EMBEDDERS_METADATA_FILEPATH, index=False, header=True)


def write_embedder_and_faiss(
    embedder_model,
    faiss_index,
    labels,
    label_encoder,
    labeled_data_df,
    validation_accuracy,
    epochs,
):
    timestamp = datetime.now()
    filename_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"classifier_model_{filename_timestamp}.pth"
    filepath = os.path.join(CLASSIFIER_CHECKPOINT_DIR, filename)
    model_state = {
        "embedder": embedder_model,
        "faiss_index": faiss.serialize_index(faiss_index),
        "labels": labels,
        "label_encoder": label_encoder,
    }
    torch.save(model_state, filepath)

    EmbedderMetadata().save_embedder_metadata(
        filename, timestamp, labeled_data_df, validation_accuracy, epochs
    )


def read_embedder_and_faiss(filepath):
    loaded_model = torch.load(filepath)
    embedder_model = loaded_model["embedder"]
    faiss_index = faiss.deserialize_index(loaded_model["faiss_index"])
    label_encoder = loaded_model["label_encoder"]
    labels = loaded_model["labels"]

    return embedder_model, faiss_index, labels, label_encoder


def read_features_for_all_images(check_data=False) -> pd.DataFrame:
    """Reads labeled and unlabeled data."""
    metadata = read_images_metadata()
    image_filepaths = metadata["filepath"]
    all_labeled_data = []
    print("read_features_for_all_images running")
    for image_filepath in image_filepaths:
        print("  ", image_filepath)
        masks_features = read_masks_features(image_filepath)

        if masks_features is None:
            print("    no masks found")
            continue

        if check_data:
            masks = read_masks_for_image(image_filepath)
            if len(masks) != masks_features.shape[0]:
                print(
                    "masks and features of {} are not compatible: {} (masks) vs {} (features)".format(
                        image_filepath, len(masks), masks_features.shape[0]
                    )
                )
                continue

        print("    adding {} masks".format(len(masks_features)))
        all_labeled_data.append(masks_features)

    print("read_features_for_all_images done")

    return pd.concat(all_labeled_data, axis=0)


def read_labeled_features_for_all_images(check_data=False) -> pd.DataFrame:
    """Reads labeled data (manually, auto, and semi-auto labeled)."""
    df = read_features_for_all_images(check_data=check_data)
    df = df[df[Y_COLUMN] != LABEL_UNLABELED]
    print("read {} labeled features (manual, auto, and semi-auto)".format(df.shape[0]))
    return df


def read_labeled_and_reviewed_features_for_all_images(check_data=False) -> pd.DataFrame:
    """Reads labeled data (manually and semi-auto labeled)."""
    df = read_features_for_all_images(check_data=check_data)
    df = df[
        (df[Y_COLUMN] != LABEL_UNLABELED)
        & (df[LABELING_MODE_COLUMN].isin([LABELING_MANUAL, LABELING_APPROVED]))
    ]
    print("read {} approved features (manual and semi-auto)".format(df.shape[0]))
    return df


def read_labels_metadata() -> pd.DataFrame:
    df = pd.read_csv(LABELS_METADATA_FILEPATH, index_col=None)
    return df
