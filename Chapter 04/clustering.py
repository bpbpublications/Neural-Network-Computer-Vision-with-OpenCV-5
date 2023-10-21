import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Read the input image
image = cv2.imread("input_images/4_Clustering.jpg")  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    sys.exit()

# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3))

# Convert the pixel values to float
pixels = np.float32(pixels)

# Define the parameters for k-means clustering
num_clusters = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply k-means clustering
_, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, flags)

# Convert the centers to 8-bit values
centers = np.uint8(centers)

# Map each pixel to its corresponding cluster center
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Convert the segmented image to RGB for visualization
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the original image and segmented image side by side
stacked_results = np.hstack((image, segmented_image_rgb))
cv2.imshow('Clustering', stacked_results)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output_images/clustering.jpg", stacked_results)