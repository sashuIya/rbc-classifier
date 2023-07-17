import numpy as np

def get_masks_img(masks, image):
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    masks_img = np.ones((image.shape[0], image.shape[1], 4))
    masks_img[:, :, 3] = 0
    for mask in sorted_masks:
        (x, y, w, h) = mask['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        segmentation = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        masks_img[y : y + h + 1, x : x + w + 1][segmentation] = color_mask

    return masks_img

def get_masked_crop(image, mask):
    (x, y, w, h) = mask['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)
    segmentation = mask['segmentation']
    crop = image[y : y + h + 1, x : x + w + 1, :]
    crop[~segmentation] = 0

    return crop