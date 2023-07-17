def is_point_in_mask(px, py, mask):
    (x, y, w, h) = mask['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)

    print(mask['segmentation'].shape)
    print(x, y, w, h)

    if px < x or px > x + w or py < y or py > y + h:
        return False

    return mask['segmentation'][py - y, px - x]