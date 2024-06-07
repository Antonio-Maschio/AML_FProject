#### Imports ####
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.ndimage import binary_erosion
from matplotlib.pyplot import imread
import skimage.io
from skimage.transform import resize

#### Functions ####
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

def load_npy_files(base_dir):
    # Define the subfolders
    subfolders = ['control', 'penetramax']
    
    # Initialize an empty list to store the outputs
    outputs = []
    
    # Loop through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        
        # Check if the subfolder exists
        if os.path.exists(subfolder_path):
            # List all files in the subfolder
            for filename in os.listdir(subfolder_path):
                # Check if the file has a .npy extension
                if filename.endswith('.npy'):
                    file_path = os.path.join(subfolder_path, filename)
                    # Load the .npy file and append the data to the outputs list
                    data = np.load(file_path)
                    outputs.append(data)
        else:
            print(f"Subfolder {subfolder_path} does not exist.")
    
    return outputs

def pad_and_resize_to_square(original_image, target_size):
    
    # Determine the larger dimension
    width, height = original_image.shape[1], original_image.shape[0]
    larger_dim  = max(width,height)
    padded_image = np.copy(original_image)
    
    if width > height:
        padded_image = np.pad(original_image, (((width-height) //2, (width-height) //2), (0,0 ), (0, 0)), mode='constant', constant_values=0)
    elif width < height :
        padded_image = np.pad(original_image, ((0,0), ( (height-width) //2, (height-width) //2), (0, 0)), mode='constant', constant_values=0)
    
    # Calculate padding and scaled dimensions
    # padding = ((larger_dim - height) // 2, (larger_dim - width) // 2)

    scaled_size = (target_size, target_size)
    scale_factor = target_size / larger_dim
    
    # Pad the image
    # padded_image = np.pad(original_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant', constant_values=0)
    
    # Resize the padded image without anti-aliasing
    resized_image = resize(padded_image, scaled_size, anti_aliasing=False)
    
    return resized_image, scale_factor

def normalize_images(image_list):
    # Separate images into two groups based on their range
    images_01 = []
    images_0255 = []

    for image in image_list:
        max_val = np.max(image)
        if max_val <= 1:
            images_01.append(image)
        else:
            images_0255.append(image)

    # Normalize images in the [0, 1] range to [0, 255]
    normalized_images_01 = [(image * 255).astype(np.uint8) for image in images_01]

    # Combine normalized images with those already in the [0, 255] range
    normalized_images = normalized_images_01 + images_0255

    return normalized_images

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        int_list = [int(num) for num in content.split(',')]
        return int_list

def remove_duplicates_and_sort(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    result.sort()
    return result