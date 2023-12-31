import os
import pickle
import cv2
import glob
import pandas as pd
import torch
import datetime
from consts import (
    Y_COLUMN,
    LABELING_MODE_COLUMN,
    LABEL_UNLABELED,
    LABELING_MANUAL,
    LABELING_APPROVED,
)
import faiss

METADATA_FILEPATH = os.path.normpath("dataset/images_metadata.csv")

CLASSIFIER_CHECKPOINT_DIR = "model/cells_classifier/"


def get_rel_filepaths_from_subfolders(folder_path, extension):
    search_pattern = folder_path + "/**/*.{}".format(extension)
    filepaths = glob.glob(search_pattern, recursive=True)
    return filepaths


def get_masks_filepath(image_filepath, suffix):
    return os.path.splitext(image_filepath)[0] + suffix + ".pkl"


def get_masks_features_filepath(image_filepath, suffix=""):
    return os.path.splitext(image_filepath)[0] + suffix + "_features.csv"


def get_classifier_model_filepaths():
    filepaths = get_rel_filepaths_from_subfolders(
        folder_path=CLASSIFIER_CHECKPOINT_DIR, extension="pth"
    )
    filepaths.sort(reverse=True)

    return filepaths


def read_masks_features(image_filepath, suffix=""):
    return pd.read_csv(
        get_masks_features_filepath(image_filepath, suffix=suffix), index_col=None
    )


def write_masks_features(
    masks_features_df: pd.DataFrame, image_filepath: str, suffix=""
):
    return masks_features_df.to_csv(
        get_masks_features_filepath(image_filepath, suffix=suffix),
        index=False,
        header=True,
    )


def read_images_metadata():
    return pd.read_csv(METADATA_FILEPATH, index_col=None)


def write_images_metadata(images_metadata: pd.DataFrame):
    return images_metadata.to_csv(METADATA_FILEPATH, index=False, header=True)


def read_image(image_filepath):
    image = cv2.imread(image_filepath)
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


def write_embedder_and_faiss(embedder_model, faiss_index, labels, label_encoder):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"classifier_model_{timestamp}.pth"
    filepath = os.path.join(CLASSIFIER_CHECKPOINT_DIR, filename)
    model_state = {
        "embedder": embedder_model,
        "faiss_index": faiss.serialize_index(faiss_index),
        "labels": labels,
        "label_encoder": label_encoder,
    }
    torch.save(model_state, filepath)


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

        if check_data:
            masks = read_masks_for_image(image_filepath)
            if len(masks) != masks_features.shape[0]:
                print(
                    "masks and features of {} are not compatible: {} (masks) vs {} (features)".format(
                        image_filepath, len(masks), masks_features.shape[0]
                    )
                )
                continue

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
