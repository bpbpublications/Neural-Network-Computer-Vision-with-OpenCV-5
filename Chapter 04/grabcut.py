import cv2
import numpy as np

# Read the image
image = cv2.imread('input_images/test_image1.jpeg')  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    exit()

# Create a mask to indicate the areas of the image to be classified (foreground, background, etc.)
mask = np.zeros(image.shape[:2], np.uint8)

# Define the rectangle enclosing the foreground object (top left and bottom right coordinates)
rect = (225, 225, 850, 850)  # Adjust the coordinates based on the region of interest

# Perform the GrabCut algorithm
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Create a mask where all probable foreground and foreground pixels are set to 1
foreground_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the original image
segmented_image = image * foreground_mask[:, :, np.newaxis]

# Display the original image and the segmented image side by side
image = cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,0), 3)
combined_image = np.hstack((image, segmented_image))
cv2.imshow('Original vs Segmented', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_images/grabcut.jpg", combined_image)