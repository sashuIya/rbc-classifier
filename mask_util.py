from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

DEVICE = "cuda"

# Define the transformation to apply to the image and mask
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the pre-trained ResNet model
resnet_model = resnet50(pretrained=True)

# Remove the classification layer from the model
features_extractor_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

# Set the model to evaluation mode
features_extractor_model.eval()


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


def save_masks(masks, image_filepath, suffix=""):
    results_filepath = os.path.splitext(image_filepath)[0] + "{}.pkl".format(suffix)
    results_filepath = os.path.normpath(results_filepath)
    print("saving to {}".format(results_filepath))
    with open(results_filepath, "wb") as f:
        pickle.dump(masks, f)


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


def get_mask_resnet_features(mask, image, model):
    bbox = [int(x) for x in mask["bbox"]]
    roi = image[bbox[1] : bbox[1] + bbox[3] + 1, bbox[0] : bbox[0] + bbox[2] + 1]
    masked_roi = roi * mask["segmentation"][:, :, np.newaxis]

    # Preprocess the image and mask
    preprocessed_roi = preprocess(masked_roi)

    # Expand the dimensions to match the input shape expected by ResNet
    preprocessed_roi = preprocessed_roi.unsqueeze(0)

    # Pass the preprocessed ROI through the model to obtain features
    with torch.no_grad():
        resnet_features = model(preprocessed_roi)

    # Flatten the feature tensor to obtain a feature vector
    feature_vector = resnet_features.view(resnet_features.size(0), -1).squeeze()

    return feature_vector.detach().cpu().numpy()
