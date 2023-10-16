import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils
from tensorflow.python.keras.layers import Dense

# Define data directory and sign language labels
data_dir = "sign_language_data"
labels = os.listdir(data_dir)

# Initialize lists to store image data and labels
images = []
encoded_labels = []

# Load and preprocess images
for label in labels:
 label_path = os.path.join(data_dir, label)
 for image_filename in os.listdir(label_path):
     image_path = os.path.join(label_path, image_filename)
     image = cv2.imread(image_path)
     image = cv2.resize(image, (64, 64)) # Resize images to a common size
     image = image / 255.0 # Normalize pixel values to [0, 1]
     images.append(image)
     encoded_labels.append(labels.index(label))

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(encoded_labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# **Fix:** The validation and test sets should be split from the same temporary set, not from the training set.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Encode labels as one-hot vectors
y_train = to_categorical(y_train, num_classes=len(labels))
y_val = to_categorical(y_val, num_classes=len(labels))
y_test = to_categorical(y_test, num_classes=len(labels))

# Display the shapes of the data splits
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)
print("Test data shape:", X_test.shape, y_test.shape)
