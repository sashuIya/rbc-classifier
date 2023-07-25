import numpy as np


def get_masks_img(masks, image):
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


def get_masked_crop(image, mask):
    (x, y, w, h) = mask["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    segmentation = mask["segmentation"]
    crop = image[y : y + h + 1, x : x + w + 1, :]
    crop[~segmentation] = 0

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
