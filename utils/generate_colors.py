import colorsys
import numpy as np

def generate_contrast_colors(num_colors):
    # Generate evenly spaced hues
    hues = np.linspace(0, 1, num_colors, endpoint=False)

    # Convert hues to RGB colors
    rgb_colors = [colorsys.hsv_to_rgb(hue, 1.0, 1.0) for hue in hues]

    # Convert RGB colors to OpenCV format
    contrast_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]

    return contrast_colors

print(generate_contrast_colors(5))
