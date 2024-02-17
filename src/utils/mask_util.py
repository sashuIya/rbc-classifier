import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from src.common.consts import (
    LABEL_UNLABELED,
    LABELING_AUTO,
    LABELING_MODE_COLUMN,
    MASK_ID_COLUMN,
)
from src.common.filepath_util import get_masks_filepath, read_images_metadata

DEVICE = "cuda"
RESNET_BATCH_SIZE = 64


def is_point_in_mask(px, py, mask):
    (x, y, w, h) = mask["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)

    if px < x or px > x + w or py < y or py > y + h:
        return False

    return mask["segmentation"][py - y, px - x]


def sam_model_version(sam_checkpoint_filepath):
    if "sam_vit_b" in sam_checkpoint_filepath:
        return "vit_b"
    if "sam_vit_h" in sam_checkpoint_filepath:
        return "vit_h"
    if "sam_vit_l" in sam_checkpoint_filepath:
        return "vit_l"

    return None


def run_sam(image, sam_checkpoint_filepath, crop_n_layers, points_per_side):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    sam = sam_model_registry[sam_model_version(sam_checkpoint_filepath)](
        checkpoint=sam_checkpoint_filepath
    )
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=16,
        # pred_iou_thresh=0.95,
        # stability_score_thresh=0.92,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        # output_mode='coco_rle',
    )

    masks = mask_generator.generate(image)
    print("found {} masks".format(len(masks)))

    # By default masks have image shape. That's too large. Crop them to corresponding bbox size.
    for mask in masks:
        x, y, w, h = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        mask["segmentation"] = mask["segmentation"][y : y + h + 1, x : x + w + 1]

    return masks


class CropsDataset(Dataset):
    def __init__(self, image, masks, transform):
        self.image = image
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = self.masks[idx]
        bbox = [int(x) for x in mask["bbox"]]
        roi = self.image[
            bbox[1] : bbox[1] + bbox[3] + 1, bbox[0] : bbox[0] + bbox[2] + 1
        ]
        masked_roi = roi * mask["segmentation"][:, :, np.newaxis]

        # Preprocess crop
        preprocessed_roi = self.transform(masked_roi)

        return preprocessed_roi


def compute_resnet_features(masks: dict, image: np.ndarray):
    """Computes resnet features for each mask and returns a pair of
    * np.ndarray of shape (n_masks, n_features)
    * column names (to construct a DataFrame)
    """
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Define the transformation to apply to the image and mask
    preprocess_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the pre-trained ResNet model
    resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Remove the classification layer from the model
    features_extractor_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    features_extractor_model.to(DEVICE)

    # Set the model to evaluation mode
    features_extractor_model.eval()

    crops_dataset = CropsDataset(image, masks, preprocess_transform)
    dataloader = DataLoader(crops_dataset, batch_size=RESNET_BATCH_SIZE, shuffle=False)
    outputs_list = []
    for batch in tqdm(
        dataloader, desc="Computing resnet features for {} crops".format(len(masks))
    ):
        batch = batch.to(DEVICE)

        with torch.no_grad():
            batch_resnet_features = features_extractor_model(batch)

        batch_resnet_features = batch_resnet_features.squeeze(-1).squeeze(-1)
        outputs_list.append(batch_resnet_features.cpu().numpy())

    all_outputs = np.concatenate(outputs_list, axis=0)

    assert all_outputs.shape == (len(masks), 2048)

    columns = ["x_resnet_{}".format(i) for i in range(2048)]

    return all_outputs, columns


def compute_measure_features(masks, image_filepath):
    """Computes measurement features of the masks. E.g., width and height
    of the bounding box in micrometers. Also returns column names (to
    construct a DataFrame).
    """
    metadata = read_images_metadata()
    micrometers, scale_x0, scale_x1 = metadata.loc[
        metadata["filepath"] == image_filepath, ["micrometers", "scale_x0", "scale_x1"]
    ].values[0]

    print(micrometers, scale_x0, scale_x1)

    scale = micrometers / abs(scale_x1 - scale_x0)

    metrics = np.zeros((len(masks), 2))
    for i, mask in enumerate(masks):
        w, h = mask["bbox"][2:4]
        metrics[i][0], metrics[i][1] = w * scale, h * scale

    return metrics, ["x_w_micrometers", "x_h_micrometers"]


def construct_features_dataframe(
    image_filepath: str, masks: dict, features: np.ndarray, feature_columns: list
):
    assert features.shape == (len(masks), len(feature_columns))
    features_df = pd.DataFrame(features, columns=feature_columns)

    metadata_df_columns = [
        "image_filepath",
        MASK_ID_COLUMN,
        "mask_area_px",
        LABELING_MODE_COLUMN,
        "y",
    ]
    metadata = []
    for mask in masks:
        metadata.append(
            [
                image_filepath,
                mask["id"],
                mask["area"],
                LABELING_AUTO,
                LABEL_UNLABELED,
            ]
        )
        assert len(metadata[-1]) == len(metadata_df_columns)

    metadata_df = pd.DataFrame(metadata, columns=metadata_df_columns)

    return pd.concat([metadata_df, features_df], axis=1)
