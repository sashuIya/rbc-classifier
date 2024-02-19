import glob
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import faiss
import numpy as np
import pandas as pd
import torch

from src.common.consts import (
    CLASSIFIER_CHECKPOINT_DIR,
    EMBEDDERS_METADATA_FILEPATH,
    IMAGES_METADATA_FILEPATH,
    INTERIM_DATA_DIR,
    LABEL_UNLABELED,
    LABELING_APPROVED,
    LABELING_MANUAL,
    LABELING_MODE_COLUMN,
    LABELS_METADATA_FILEPATH,
    RAW_IMAGES_DIR,
    Y_COLUMN,
)


def get_rel_filepaths_from_subfolders(folder_path, extension, exclude=None):
    folder_path = Path(folder_path)
    search_pattern = folder_path.glob(f"**/*.{extension}")

    filepaths = list(search_pattern)
    if exclude is not None:
        filepaths = [filepath for filepath in filepaths if exclude not in str(filepath)]

    return sorted(filepaths)


def _get_interim_folder_path(image_filepath: Path) -> Path:
    folder_path = (
        INTERIM_DATA_DIR
        / image_filepath.parent.relative_to(RAW_IMAGES_DIR)
        / image_filepath.stem
    )

    return folder_path


def _get_masks_folder_path(image_filepath: str) -> Path:
    image_filepath = Path(image_filepath)
    return _get_interim_folder_path(image_filepath) / "masks"


def _get_masks_features_folder_path(image_filepath: str):
    image_filepath = Path(image_filepath)
    return _get_interim_folder_path(image_filepath) / "mask_features"


def get_classifier_model_filepaths():
    filepaths = get_rel_filepaths_from_subfolders(
        folder_path=CLASSIFIER_CHECKPOINT_DIR, extension="pth"
    )
    filepaths = sorted(filepaths, key=lambda x: x.stat().st_ctime, reverse=True)

    return filepaths


class ImageDataReader:
    def __init__(self, image_filepath: str):
        self.image_filepath = Path(image_filepath)
        self.masks_folder_path = _get_masks_folder_path(image_filepath)
        self.masks_features_folder_path = _get_masks_features_folder_path(
            image_filepath
        )

    def masks_options(self) -> List[str]:
        mask_features_filepaths = get_rel_filepaths_from_subfolders(
            self.masks_features_folder_path, "csv"
        )

        # Sort the list by creation date, most recent first.
        mask_features_filepaths = sorted(
            mask_features_filepaths, key=lambda x: x.stat().st_ctime, reverse=True
        )

        return [f.stem for f in mask_features_filepaths]

    def read_masks(self, masks_option: str) -> Dict:
        masks_filepath = self.masks_folder_path / (masks_option + ".pkl")
        if not masks_filepath.exists():
            print(f"{masks_filepath} does not exist")
            return None

        with open(masks_filepath, "rb") as f:
            masks = pickle.load(f)

        sorted_masks = sorted(masks, key=(lambda x: x["area"]))
        for i, mask in enumerate(sorted_masks):
            mask["id"] = i

        return sorted_masks

    def read_masks_features(self, masks_option: str) -> pd.DataFrame:
        masks_features_filepath = self.masks_features_folder_path / (
            masks_option + ".csv"
        )

        if not masks_features_filepath.exists():
            return None

        return pd.read_csv(masks_features_filepath, index_col=None)

    def read_latest_masks(self):
        masks_options = self.masks_options()
        if len(masks_options) == 0:
            return None

        return self.read_masks(masks_options[0])

    def read_latest_masks_features(self):
        masks_options = self.masks_options()
        if len(masks_options) == 0:
            return None

        return self.read_masks_features(masks_options[0])


class ImageDataWriter:
    def __init__(self, image_filepath: str):
        self.image_filepath = Path(image_filepath)
        self.masks_folder_path = _get_masks_folder_path(image_filepath)
        self.masks_features_folder_path = _get_masks_features_folder_path(
            image_filepath
        )

    def write_masks(self, masks, name: str):
        masks_filepath = self.masks_folder_path / (name + ".pkl")

        if not masks_filepath.parent.exists():
            masks_filepath.parent.mkdir(parents=True, exist_ok=True)

        print(f"saving to {masks_filepath}")
        with masks_filepath.open("wb") as f:
            pickle.dump(masks, f)

    def write_masks_features(self, masks_features_df: pd.DataFrame, name: str):
        mask_features_filepath = self.masks_features_folder_path / (name + ".csv")
        if not mask_features_filepath.parent.exists():
            mask_features_filepath.parent.mkdir(parents=True, exist_ok=True)

        masks_features_df.to_csv(
            mask_features_filepath,
            index=False,
            header=True,
        )


def read_images_metadata():
    if not os.path.exists(LABELS_METADATA_FILEPATH):
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
        image_data_reader = ImageDataReader(image_filepath)
        for masks_option in image_data_reader.masks_options():
            masks_features = image_data_reader.read_masks_features(masks_option)

            if masks_features is None:
                print("    no masks found")
                continue

            if check_data:
                masks = image_data_reader.read_masks(masks_option)
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
