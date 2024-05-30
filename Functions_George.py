#### Imports ####
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.pyplot import imshow, show, subplot,title,axis
from matplotlib.patches import Circle
from skimage.io import imread
import skimage.util
import skimage as sk
from timeit import timeit
from skimage.feature import corner_peaks,peak_local_max
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from skimage import exposure, filters, morphology
from cellpose import models, io, denoise, utils
from cellpose.io import imread, logger_setup
import pandas as pd
from skimage.measure import regionprops, label, regionprops_table
from skimage.feature import blob_log
from skimage.exposure import rescale_intensity
from skimage import transform
from skimage.draw import polygon_perimeter
from scipy.ndimage import binary_erosion

import os

# Add OpenMP runtime conflict resolution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow program to continue despite multiple OpenMP runtimes
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to one to avoid conflicts

def compute_cdf(histogram):
    # Compute the cumulative sum of the histogram
    cumulative_histogram = np.cumsum(histogram)
    # Normalize the cumulative histogram to obtain the CDF
    cdf = cumulative_histogram / np.sum(histogram)
    return cdf

def C(image, cdf):
    c = cdf[image]
    return c

def custom_colormap():
    n_intervals = 10
    intervals = np.linspace(0, 1, n_intervals)
    cmap_dict = {'red': [], 'green': [], 'blue': []}
    for interval in intervals:
        r = 1  # Max intensity (red)
        g = 0  # Min intensity (green)
        b = 0  # Min intensity (blue)
        cmap_dict['red'].append((interval, r, r))
        cmap_dict['green'].append((interval, g, g))
        cmap_dict['blue'].append((interval, b, b))
    custom_cmap = LinearSegmentedColormap('custom', cmap_dict)
    return custom_cmap

def detect_outline(image):
    
    image = image.astype(bool)
    struct = np.ones((3, 3), dtype=bool)
    eroded_image = binary_erosion(image, structure=struct)
    outline = image ^ eroded_image
    
    return outline

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {filename}")
    return images