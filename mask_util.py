from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

DEVICE = 'cuda'

def is_point_in_mask(px, py, mask):
    (x, y, w, h) = mask['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)

    if px < x or px > x + w or py < y or py > y + h:
        return False

    return mask['segmentation'][py - y, px - x]

def sam_model_version(sam_checkpoint_filepath):
    if 'sam_vit_b' in sam_checkpoint_filepath:
        return 'vit_b'
    if 'sam_vit_h' in sam_checkpoint_filepath:
        return 'vit_h'
    if 'sam_vit_l' in sam_checkpoint_filepath:
        return 'vit_l'
    
    return None

def run_sam(image, sam_checkpoint_filepath):
    sam = sam_model_registry[sam_model_version(sam_checkpoint_filepath)](checkpoint=sam_checkpoint_filepath)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=1,
        # pred_iou_thresh=0.95,
        # stability_score_thresh=0.92,
        # crop_n_layers=0,
        # crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',
    )

    masks = mask_generator.generate(image)
    print('found {} masks'.format(len(masks)))

    return masks
