#### Imports ####
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.pyplot import imshow, show, subplot,title,axis
from matplotlib.patches import Circle
from skimage.io import imread
from skimage import img_as_float,img_as_ubyte
from skimage.color import rgb2hsv,hsv2rgb,rgb2gray
import skimage.util
import skimage as sk
import scipy
import scipy.ndimage
import scipy.signal
import scipy.signal as signal
from scipy.signal import convolve,gaussian
from scipy.signal import convolve2d
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import ndimage
import matplotlib.patches as patches
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy,binary_crossentropy
from keras.optimizers import Adadelta,Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch,BayesianOptimization
from sklearn.model_selection import KFold


## Load whole images 
def load_images_from_folder(folder_path):
    """
    Load all images from the specified folder and return them as a list of images read by OpenCV.
    """
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {filename}")
    return images

# Read masks
def load_npy_files(base_dir):
    subfolders = ['control', 'penetramax']
    
    outputs = []
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.exists(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(subfolder_path, filename)
                    data = np.load(file_path)
                    outputs.append(data)
        else:
            print(f"Subfolder {subfolder_path} does not exist.")
    return outputs

def get_masks(base_path):
    segments = load_npy_files(base_path)
    segments_nuclei = []
    for segment in segments:
        segments_nuclei.append(segment[:, :, 1])
    return segments_nuclei[:10],segments_nuclei[10:]

# Find median image
def get_median_image(ctrl_masks,pntr_masks):
    from skimage import measure
    ctrl_temp = []
    pntr_temp = []
    for k,i in enumerate(ctrl_masks):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            bi_mask = reg.image
            ctrl_temp.append(bi_mask.shape)
    for k,i in enumerate(pntr_masks):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            bi_mask = reg.image
            pntr_temp.append(bi_mask.shape)

    ctrl_temp = np.array(ctrl_temp)
    pntr_temp = np.array(pntr_temp)
    
    max_ctrl = np.median(ctrl_temp, axis=0)
    max_pntr = np.median(pntr_temp, axis=0)
    
    return int(np.median([max_ctrl, max_pntr]))

#Get each cell in an image list
def get_cells_re(image_list,mask_list):
    from skimage import measure
    temp = []
    for k,i in enumerate(mask_list):
        regions = measure.regionprops(i)
        for j,reg in enumerate(regions):
            min_row, min_col, max_row, max_col = reg.bbox

            region_of_interest = image_list[k][min_row:max_row,min_col:max_col]
            
            temp.append(region_of_interest)
    return temp

#Plot all images 6x6
def plot_all(image_list,until):
    images = image_list[:until]
    binary_nucleus_images=[]
    for i,im in enumerate(images):
        binary_nucleus_images.append(im)
            # If we have accumulated 6 images, display them
        if len(binary_nucleus_images) == 6:

            fig, ax = plt.subplots(1, 6, figsize=(18, 3)) 
            
            for o in range(6):
                ax[o].imshow(binary_nucleus_images[o], cmap='gray')
                ax[o].axis('off')  # Hide the axes for better visualization
            fig.text(0.0, 0.5, f'Nucleus Numbers from {i - 5} to {i}', va='center', rotation='vertical', fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # Clear the list for the next set of images
            binary_nucleus_images.clear()

#Get the indices to throw in np.arrays
def get_dis():
    with open('ctrl_dis.txt', 'r') as file:
        text_values_ctrl = file.read().strip()
    values_list_ctrl = text_values_ctrl.split(',')
    values_list_ctrl = [int(value) for value in values_list_ctrl]
    ctrl_dis= np.array(values_list_ctrl)

    with open('pntr_dis.txt', 'r') as file:
        text_values_pntr = file.read().strip()
    values_list_pntr = text_values_pntr.split(',')
    values_list_pntr = [int(value) for value in values_list_pntr]
    pntr_dis= np.array(values_list_pntr)
    return ctrl_dis,pntr_dis

#Discard unwanted images
def discard(image_list,indices):
    indices_to_remove_set_ctrl = set(indices)

    cell_images = [image for idx, image in enumerate(image_list) if idx not in indices_to_remove_set_ctrl]
    return cell_images

#Chech if list has 0-value images for list
def has_zeros_list(image_list):
    zero_arrays_count = sum(np.all(arr == 0) for arr in image_list)
    print(f"Number of arrays with all zero values: {zero_arrays_count}")

#Chech if list has 0-value images for np.array
def has_zeros_array(image_array):
    zero_arrays_count = np.sum(np.all(image_array == 0, axis=(1, 2, 3)))
    print(f"Nuber of arrays with all zero values: {zero_arrays_count}")


def pad_and_resize_to_square(original_image, target_size):
    import numpy as np
    from skimage.transform import resize
    
    # Determine the original width and height
    height, width = original_image.shape[:2]
    
    # Calculate padding for width and height
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = (width - height) - pad_top
        pad_left = pad_right = 0
    else:
        pad_left = (height - width) // 2
        pad_right = (height - width) - pad_left
        pad_top = pad_bottom = 0
    
    # Apply padding
    padded_image = np.pad(original_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    
    # Resize the padded image to the target size
    resized_image = resize(padded_image, (target_size, target_size), anti_aliasing=False)
    
    # Calculate the scale factor
    larger_dim = max(width, height)
    scale_factor = target_size / larger_dim
    
    return resized_image, scale_factor


def image_to_hash(image):
    import hashlib
    """Convert image to a hash to ensure uniqueness."""
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

# Function to process images and add to respective lists if unique
def process_and_add_images(images):
    image_hashes = set()
    new_rot = []
    new_flip = []
    for i in images:
        # Original image hash
        original_hash = image_to_hash(i)
        if original_hash not in image_hashes:
            image_hashes.add(original_hash)
            new_rot.append(i)

        # Rotate images and append to the new list if unique
        for angle in [90, 180, 270]:
            rotated_image = skimage.transform.rotate(np.copy(i), angle=angle, resize=True, preserve_range=True).astype(i.dtype)
            rotated_hash = image_to_hash(rotated_image)
            if rotated_hash not in image_hashes:
                image_hashes.add(rotated_hash)
                new_rot.append(rotated_image)
        
        # Flip images and append to the new list if unique
        flipped_lr = np.fliplr(np.copy(i)).astype(i.dtype)
        flipped_ud = np.flipud(np.copy(i)).astype(i.dtype)
        flipped_lr_hash = image_to_hash(flipped_lr)
        flipped_ud_hash = image_to_hash(flipped_ud)
        
        if flipped_lr_hash not in image_hashes:
            image_hashes.add(flipped_lr_hash)
            new_flip.append(flipped_lr)
        
        if flipped_ud_hash not in image_hashes:
            image_hashes.add(flipped_ud_hash)
            new_flip.append(flipped_ud)
    
    return new_rot+new_flip

#Normalize images from 0 to 255 if they are not
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

#Check if they have the same range
def check_image_ranges(image_list):
    ranges = set()
    for image in image_list:
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > 1:
            ranges.add("0255")
        else:
            ranges.add("01")
    
    if len(ranges) == 1:
        if "01" in ranges:
            return "All images are within the range [0, 1]."
        elif "0255" in ranges:
            return "All images are within the range [0, 255]."
    else:
        return "Images have mixed ranges."

#Suffle arrays and split to training,validation,test
def shuffle_and_split(images,factors,labels):
    np.random.seed(seed=49999)
    indices = np.random.permutation(len(images))
    shuffled_images = images[indices]
    shuffled_scaling_factors = factors[indices]
    shuffled_labels = labels[indices]

    # Total number of images
    total_images = len(shuffled_images)
    # Split indices for 80%, 10%, 10% split
    train_split_index = int(total_images * 0.8)
    validation_split_index = int(total_images * 0.9)
    
    # Splitting into training, validation, and testing sets
    train_images = shuffled_images[:train_split_index]
    train_scaling_factors = shuffled_scaling_factors[:train_split_index]
    train_labels = shuffled_labels[:train_split_index]

    validation_images = shuffled_images[train_split_index:validation_split_index]
    validation_scaling_factors = shuffled_scaling_factors[train_split_index:validation_split_index]
    validation_labels = shuffled_labels[train_split_index:validation_split_index]

    test_images = shuffled_images[validation_split_index:]
    test_scaling_factors = shuffled_scaling_factors[validation_split_index:]
    test_labels = shuffled_labels[validation_split_index:]

    return train_images,train_scaling_factors,train_labels,validation_images,validation_scaling_factors,validation_labels,test_images,test_scaling_factors,test_labels

path_ctrl = r"C:\Users\dimea\OneDrive\Documents\University Courses\B.4 Applied Machine Learning\Final project\control-images"
path_pntr = r"C:\Users\dimea\OneDrive\Documents\University Courses\B.4 Applied Machine Learning\Final project\penetrax-images"
ctrl_im = load_images_from_folder(path_ctrl)
pntr_im = load_images_from_folder(path_pntr)

ctrl_mask_list,pntr_mask_list = get_masks("segments")

ctrl_cell_images_bef  = get_cells_re(ctrl_im,ctrl_mask_list)
pntr_cell_images_bef  = get_cells_re(pntr_im,pntr_mask_list)

ctrl_dis,pntr_dis = get_dis()

print("Number of control cells:",len(ctrl_cell_images_bef))
print("Number of penetramax cells:",len(pntr_cell_images_bef))

ctrl_cell_images = discard(ctrl_cell_images_bef,ctrl_dis)
pntr_cell_images = discard(pntr_cell_images_bef,pntr_dis)

print("Number of control cells after filtering:",len(ctrl_cell_images))
print("Number of penetramax cells after filtering",len(pntr_cell_images))

median_image = get_median_image(ctrl_mask_list,pntr_mask_list)
print("Size of median image: ",median_image)

pntr_new = process_and_add_images(ctrl_cell_images)
ctrl_new = process_and_add_images(pntr_cell_images)

ctrl_cell_images.extend(pntr_new)
pntr_cell_images.extend(ctrl_new)

num_ctrl_cell_images = len(ctrl_cell_images)
num_pntr_cell_images = len(pntr_cell_images)

print("Total unique control cell images:", len(ctrl_cell_images))
print("Total unique pointer cell images:", len(pntr_cell_images))

all_images = ctrl_cell_images+pntr_cell_images

# Initialize lists to store results
resized_images_list = []
scaling_factors_list = []

# Process each image
for i in range(len(all_images)):
    resized_image, scale_factor = pad_and_resize_to_square(all_images[i], median_image)
    resized_images_list.append(resized_image)
    scaling_factors_list.append(scale_factor)

ctrl_cell_images_norm = normalize_images(resized_images_list)

# Convert lists to numpy arrays
all_images_resized = np.array(ctrl_cell_images_norm).astype(int)
scaling_factors = np.array(scaling_factors_list)

ctrl_labels = np.zeros(num_ctrl_cell_images)
pntr_labels = np.ones(num_pntr_cell_images)
all_labels = np.concatenate((ctrl_labels, pntr_labels), axis=0)

X_train,fact_train,y_train,X_val,fact_val,y_val,X_test,fact_test,y_test = shuffle_and_split(all_images_resized,scaling_factors,all_labels)
print("Training images shape:", X_train.shape)
print("Training scaling factors shape:", fact_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation images shape:", X_val.shape)
print("Validation scaling factors shape:", fact_val.shape)
print("Validation labels shape:", y_val.shape)
print("Testing images shape:", X_test.shape)
print("Testing scaling factors shape:", fact_test.shape)
print("Testing labels shape:", y_test.shape)

X_training = X_train/255.
X_validation = X_val/255.
X_testing = X_test/255.

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, scales_shape):
        self.input_shape = input_shape
        self.scales_shape = scales_shape

    def build(self, hp):
        image_input = Input(shape=self.input_shape, name='image_input')
        scales_input = Input(shape=self.scales_shape, name='scales_input')

        # Determine the number of convolutional layers
        num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=5)

        x = image_input
        for i in range(num_conv_layers):
            x = Conv2D(
                filters=hp.Int(f'conv_{i+1}_filters', min_value=48, max_value=512, step=16),
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            )(x)
            x = MaxPooling2D(pool_size=2)(x)
        
        x = Flatten()(x)

        x = Lambda(lambda inputs: K.concatenate([inputs[0], inputs[1]], axis=-1))([x, scales_input])

        # Determine the number of dense layers
        num_dense_layers = hp.Int('num_dense_layers', min_value=3, max_value=10)

        for i in range(num_dense_layers):
            units = hp.Int(f'dense_{i+1}_units', min_value=100, max_value=2000)
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(rate=hp.Float(f'dropout_{i+1}_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
        
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[image_input, scales_input], outputs=output)

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')),
            metrics=['accuracy']
        )

        return model

input_shape = (median_image, median_image, 3)
scales_shape = (1,)

hypermodel = CNNHyperModel(input_shape=input_shape, scales_shape=scales_shape)

tuner = BayesianOptimization(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='dir',
    project_name='hyperparam_tuning'
)

tuner.search(
    [X_training, fact_train], y_train,
    epochs=10,
    validation_data=([X_validation, fact_val], y_val),
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of convolutional layers is {best_hps.get('num_conv_layers')}.
The optimal number of dense layers is {best_hps.get('num_dense_layers')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

image_input = Input(shape=(median_image, median_image, 3), name='image_input')

scales_input = Input(shape=(1,), name='scales_input')

# Convolutional layers
x = Conv2D(48, (3, 3), strides=1, padding='same', activation='relu')(image_input)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)

x = Lambda(lambda inputs: K.concatenate([inputs[0], inputs[1]], axis=-1))([x, scales_input])

# Fully connected layers
x = Dense(768, activation='relu')(x)
x = Dense(1536, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(768, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(384, activation='relu')(x)
x = Dense(192, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[image_input, scales_input], outputs=output)

model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

history = model.fit([X_training, fact_train], y_train,
                    epochs=10,
                    validation_data=([X_validation, fact_val], y_val),
                    verbose=1)


history_dict = history.history

plt.figure(figsize=(12, 6))

epochs = range(1, len(history_dict['loss']) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict['loss'], 'bo-', label='Training loss')
plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict['accuracy'], 'bo-', label='Training accuracy')
plt.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

predictions = model.predict([X_testing,fact_test])

print(accuracy_score(y_test,predictions.round()))

conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predictions.round(), num_classes=2)
print('Confusion Matrix: ', conf_matrix)

conf_matrix_np = conf_matrix.numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_np, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

X_data = all_images_resized/255.
fact_data = scaling_factors
y_data = all_labels

# Initialize K-Fold Cross-Validation
k = 8
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Store results
scores = []

# Cross-validation process
for train_index, val_index in kf.split(X_data):

    X_train, X_val = X_data[train_index], X_data[val_index]
    fact_train, fact_val = fact_data[train_index], fact_data[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]
    
    
    # Train the model
    history = model.fit([X_train, fact_train], y_train,
                        epochs=15,
                        validation_data=([X_val, fact_val], y_val),
                        verbose=1)
    
    # Evaluate the model
    score = model.evaluate([X_val, fact_val], y_val, verbose=0)
    scores.append(score[1])  # Assuming score[1] is accuracy

# Calculate the average accuracy
average_accuracy = np.mean(scores)
print(f'Average accuracy: {average_accuracy}')
