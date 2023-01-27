import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
import cv2

# Load training images
training_images = []
for i in range(1, 6):
    image = cv2.imread(f'faces/training/{i}.jpg', 0)
    training_images.append(image)
training_images = np.array(training_images)

# Load test image
test_image = cv2.imread('faces/testing/sample.jpg', 0)

# Flatten images
training_images = training_images.reshape(training_images.shape[0], -1)
test_image = test_image.reshape(1, -1)

# Perform PCA on training images
pca = PCA(n_components=0.95)
pca.fit(training_images)

# Project training images and test image onto the PCA space
projected_training_images = pca.transform(training_images)
projected_test_image = pca.transform(test_image)

# Define threshold
threshold = 50

# Compare distances between projected test image and projected training images
distances = euclidean_distances(projected_test_image, projected_training_images)
min_distance = np.min(distances)

# Check if test image is recognized
if min_distance < threshold:
    print("Test image recognized.")
else:
    print("Test image not recognized.")