import base64
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import cv2
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html

from src.common.consts import (
    ALL_MASKS_RADIO_BUTTONS_PREFIX,
    CONFIDENCE_COLUMN,
    LABELING_MODE_COLUMN,
    MASK_ID_COLUMN,
    Y_COLUMN,
)
from src.common.filepath_util import LabelsMetadata
from src.utils.draw_util import get_masked_crop


@dataclass
class LabeledMaskPreviewInfo:
    highlighted_crop: np.ndarray
    original_crop: np.ndarray
    label: str
    labeling_mode: str
    mask: dict
    confidence_score: float


def _get_confidence_score(labels_df: pd.DataFrame, mask_id: int) -> float:
    if CONFIDENCE_COLUMN not in labels_df.columns:
        return 0.0

    return labels_df[labels_df[MASK_ID_COLUMN] == mask_id][CONFIDENCE_COLUMN].values[0]


def crop_html(crop):
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
    _, crop_png = cv2.imencode(".png", bgr_crop)
    crop_base64 = base64.b64encode(crop_png).decode("utf-8")
    crop_html = html.Img(
        src=f"data:image/png;base64,{crop_base64}",
        style={"display": "block", "margin-bottom": "10px"},
        className="img-item",
    )

    return crop_html


def generate_labeled_mask_preview_info(
    image: np.array,
    mask: dict,
    label: str,
    labeling_mode: str,
    confidence_score: float,
    labels_metadata: LabelsMetadata,
) -> LabeledMaskPreviewInfo:
    """Generates a preview of the given `mask` labeled with the given `label`.

    Args:
        image (np.array): Image to crop.
        mask (dict): A mask from SAM.
        label (str): label of the mask.
        confidence_score (float): Confidence score of the label.

    Returns:
        LabeledMaskPreview: An instance containing (highlighted_crop, original_crop, label, mask, confidence_score).
    """
    highlighted_crop = get_masked_crop(
        image,
        mask,
        xy_threshold=20,
        with_highlighting=True,
        color_mask=labels_metadata.get_color_by_label(label),
    )
    original_crop = get_masked_crop(
        image,
        mask,
        xy_threshold=20,
        with_highlighting=False,
        color_mask=labels_metadata.get_color_by_label(label),
    )
    return LabeledMaskPreviewInfo(
        highlighted_crop=highlighted_crop,
        original_crop=original_crop,
        label=label,
        labeling_mode=labeling_mode,
        mask=mask,
        confidence_score=confidence_score,
    )


def generate_crop_with_radio(
    id_prefix: str,
    labeled_mask_preview_info: LabeledMaskPreviewInfo,
    labels: List[str],
) -> dbc.Form:
    mask = labeled_mask_preview_info.mask
    mask_id = mask["id"]
    return dbc.Form(
        [
            dbc.Row(
                html.Div(
                    "mask_id: {}, labeling_mode: {}, area: {}, confidence_score: {}".format(
                        mask_id,
                        labeled_mask_preview_info.labeling_mode,
                        mask["area"],
                        labeled_mask_preview_info.confidence_score,
                    ),
                    id={
                        "type": id(f"{id_prefix}-mask-id-div"),
                        "index": mask_id,
                    },
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        crop_html(labeled_mask_preview_info.highlighted_crop),
                        className="horizontal-space",
                        style={
                            "width": "{}px".format(
                                labeled_mask_preview_info.highlighted_crop.shape[1]
                            ),
                            "margin-right": "20px",
                        },
                        width="auto",
                    ),
                    dbc.Col(
                        crop_html(labeled_mask_preview_info.original_crop),
                        className="horizontal-space",
                        style={
                            "width": "{}px".format(
                                labeled_mask_preview_info.original_crop.shape[1]
                            ),
                            "margin-right": "20px",
                        },
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            options=labels,
                            value=labeled_mask_preview_info.label,
                            id={
                                "type": id(f"{id_prefix}-radio-item"),
                                "index": mask_id,
                            },
                            inline=False,
                            className="ml-3",
                        )
                    ),
                ],
                className="img-container",
                style={"margin-bottom": "5px"},
            ),
        ],
        className="labeled-mask-card",
    )


def generate_labeled_masks_previews(
    radio_buttons_prefix: str,
    labeled_mask_preview_infos: List[LabeledMaskPreviewInfo],
    labels_metadata: LabelsMetadata,
) -> List[dbc.Form]:
    labels = list(labels_metadata.get_list_of_labels())

    image_radio_items = []
    for info in labeled_mask_preview_infos:
        image_radio_item = generate_crop_with_radio(
            id_prefix=radio_buttons_prefix,
            labeled_mask_preview_info=info,
            labels=labels,
        )
        image_radio_items.append(image_radio_item)

    return image_radio_items


def generate_rows_of_mask_previews(
    image: np.ndarray,
    labeled_masks_df: pd.DataFrame,
    masks: List[dict],
    labels_metadata: LabelsMetadata,
    radio_buttons_prefix: str,
) -> List[dbc.Row]:
    crop_infos: List[LabeledMaskPreviewInfo] = []
    for (_, row), mask in zip(labeled_masks_df.iterrows(), masks):
        assert mask["id"] == row[MASK_ID_COLUMN], (mask["id"], row[MASK_ID_COLUMN])
        confidence_score = _get_confidence_score(labeled_masks_df, mask["id"])
        crop_infos.append(
            generate_labeled_mask_preview_info(
                image,
                mask,
                row[Y_COLUMN],
                row[LABELING_MODE_COLUMN],
                confidence_score,
                labels_metadata,
            )
        )

    sorted_crop_infos = sorted(
        crop_infos, key=lambda crop_info: crop_info.confidence_score
    )

    grouped_by_label = defaultdict(list)
    for crop_info in sorted_crop_infos:
        grouped_by_label[crop_info.label].append(crop_info)

    rows: List[dbc.Row] = []
    for crop_infos in grouped_by_label.values():
        mask_previews = generate_labeled_masks_previews(
            radio_buttons_prefix, crop_infos, labels_metadata
        )
        rows.append(
            dbc.Row(
                [
                    dbc.Col(mask_preview, style={"padding": "0px"})
                    for mask_preview in mask_previews
                ],
                className="masks-class-preview-row",
            ),
        )

    return rows
