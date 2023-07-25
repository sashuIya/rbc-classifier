import os
import pickle
import cv2
import glob
import pandas as pd

LABEL_UNLABELED = "unlabeled"
METADATA_FILEPATH = os.path.normpath("dataset/images_metadata.csv")


def get_rel_filepaths_from_subfolders(folder_path, extension):
    tif_files = []
    search_pattern = folder_path + "/**/*.{}".format(extension)
    tif_files = glob.glob(search_pattern, recursive=True)
    return tif_files


def get_masks_filepath(image_filepath, suffix):
    return os.path.splitext(image_filepath)[0] + suffix + ".pkl"


def get_labels_filepath(image_filepath):
    return os.path.splitext(image_filepath)[0] + "_labels.csv"


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


def write_labels(image_filepath: str, labels: pd.DataFrame):
    labels_filepath = get_labels_filepath(image_filepath)
    labels.to_csv(labels_filepath, header=True, index=False)


def read_labels_for_image(image_filepath):
    labels_filepath = get_labels_filepath(image_filepath)
    if not os.path.exists(labels_filepath):
        masks = read_masks_for_image(image_filepath)

        data = {
            "id": [mask["id"] for mask in masks],
            "label": [LABEL_UNLABELED] * len(masks),
            "manually_labeled": [False] * len(masks),
        }

        combined_labels = pd.DataFrame(data)
        write_labels(image_filepath, combined_labels)

    labels = pd.read_csv(labels_filepath, index_col=False)
    return labels
