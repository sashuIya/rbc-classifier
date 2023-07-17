import os
import pickle
import cv2

def get_masks_filepath(image_filepath):
    return os.path.splitext(image_filepath)[0] + '.pkl'

def read_image(image_filepath):
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_masks(masks_filepath):
    with open(masks_filepath,'rb') as f:
        masks = pickle.load(f)
    return masks

def read_masks_for_image(image_filepath):
    masks_filepath = get_masks_filepath(image_filepath)
    with open(masks_filepath,'rb') as f:
        masks = pickle.load(f)
    return masks

