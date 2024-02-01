from enum import Enum
import random
import numpy as np

from utils.generate_colors import generate_contrast_colors


class MasksColorOptions(Enum):
    NONE = 1  # do not display masks
    RANDOM = 2  # display each mask with random color
    BY_LABEL = 3  # color is based on mask's label


def get_masks_img(
    masks,
    image,
    masks_color_option=MasksColorOptions.BY_LABEL,
    color_by_mask_id=None,
    opacity=0.35,
):
    if masks_color_option == MasksColorOptions.NONE:
        return image

    if masks_color_option == MasksColorOptions.BY_LABEL and color_by_mask_id is None:
        raise ValueError(
            "Color palette is not provided for labeled data display option"
        )

    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    if masks_color_option == MasksColorOptions.RANDOM:
        colors = generate_contrast_colors(len(masks))
        random.shuffle(colors)

    for index, mask in enumerate(sorted_masks):
        mask_id = mask["id"]
        (x, y, w, h) = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        segmentation = mask["segmentation"]
        if masks_color_option == MasksColorOptions.BY_LABEL:
            if mask_id not in color_by_mask_id:
                # Skip masks without defined color (e.g., if it's labeled as wrong).
                continue
            mask_color = np.array(color_by_mask_id[mask_id])
        elif masks_color_option == MasksColorOptions.RANDOM:
            mask_color = np.array(colors[index])
        else:
            raise ValueError("Invalid color")

        mask_region = image[y : y + h + 1, x : x + w + 1][segmentation]
        blended_color = ((1 - opacity) * mask_region + opacity * mask_color).astype(int)
        image[y : y + h + 1, x : x + w + 1][segmentation] = blended_color

    return image


def get_masks_img_old(masks, image):
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    masks_img = np.ones((image.shape[0], image.shape[1], 4))
    masks_img[:, :, 3] = 0
    for mask in sorted_masks:
        (x, y, w, h) = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        segmentation = mask["segmentation"]
        color_mask = (
            mask["color"]
            if "color" in mask
            else np.concatenate([np.random.random(3), [0.35]])
        )
        masks_img[y : y + h + 1, x : x + w + 1][segmentation] = color_mask

    return masks_img


def get_masked_crop(
    image,
    mask,
    xy_threshold,
    with_highlighting,
    color_mask=np.array([0, 0, 255]),
    opacity=0.35,
):
    (x, y, w, h) = mask["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    segmentation = mask["segmentation"]

    y0 = max(0, y - xy_threshold)
    x0 = max(0, x - xy_threshold)
    y1 = min(image.shape[0], y + xy_threshold + h + 1)
    x1 = min(image.shape[1], x + xy_threshold + w + 1)
    crop = np.copy(image[y0:y1, x0:x1, :])
    alpha_channel = np.full((crop.shape[0], crop.shape[1], 1), 255, dtype=np.uint8)
    crop = np.concatenate((crop, alpha_channel), axis=2)

    segmentation_threshold = np.zeros(crop.shape[:2], dtype=bool)
    segmentation_threshold[y - y0 : y - y0 + h + 1, x - x0 : x - x0 + w + 1] = (
        segmentation
    )

    if with_highlighting:
        mask_region = crop[segmentation_threshold, :3]
        blended_color = ((1 - opacity) * mask_region + opacity * color_mask).astype(int)
        crop[segmentation_threshold, :3] = blended_color

    return crop


def draw_height_and_scale(fig, image_shape, height, scale_x0, scale_x1, micrometers):
    x = [image_shape[1] // 3] * 2
    y = [0, height]
    fig.add_scatter(x=x, y=y, line=dict(color="#ff0076"))

    x = [scale_x0, scale_x1]
    y = [height, height]
    fig.add_scatter(x=x, y=y, line=dict(color="#ff0076"))

    fig.add_annotation(
        dict(
            x=(scale_x0 + scale_x1) // 2,
            y=height,
            text="{} micrometers".format(micrometers),
            textangle=0,
            font=dict(color="red", size=24),
        )
    )

    fig.add_annotation(
        dict(
            x=image_shape[1] // 3 - 50,
            y=height // 2,
            text="{}px".format(height),
            textangle=-90,
            font=dict(color="red", size=24),
        )
    )
