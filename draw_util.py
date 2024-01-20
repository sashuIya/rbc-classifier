import numpy as np


def get_masks_img(masks, image, opacity=0.35):
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    for mask in sorted_masks:
        (x, y, w, h) = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        segmentation = mask["segmentation"]
        color_mask = (
            np.array(mask["color"])
            if "color" in mask
            else np.random.randint(256, size=3)
        )

        mask_region = image[y : y + h + 1, x : x + w + 1][segmentation]
        blended_color = ((1 - opacity) * mask_region + opacity * color_mask).astype(int)
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


def get_masked_crop(image, mask, xy_threshold, with_highlighting):
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
    segmentation_threshold[
        y - y0 : y - y0 + h + 1, x - x0 : x - x0 + w + 1
    ] = segmentation

    # darkening_factor = 20
    # crop[segmentation_threshold, :3] -= darkening_factor
    # crop[:, :, :3] = np.clip(crop[:, :, :3], 0, 255)

    if with_highlighting:
        crop[segmentation_threshold, 0] = np.clip(
            crop[segmentation_threshold, 0] * 1.2, 0, 255
        ).astype(np.uint8)
        crop[segmentation_threshold, 1] = np.clip(
            crop[segmentation_threshold, 1] * 1.3, 0, 255
        ).astype(np.uint8)
        crop[segmentation_threshold, 2] = np.clip(
            crop[segmentation_threshold, 1] * 1.4, 0, 255
        ).astype(np.uint8)

        crop[:, :, 3] = 150
        crop[segmentation_threshold, 3] = 255

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
