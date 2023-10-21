import cv2
import numpy as np

# Read the input image
image = cv2.imread("input_images/4_ThresholdingImage.jpg")  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    exit()

# Define the structuring element for erosion
kernel_size = 5
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# Perform image erosion
eroded_image = cv2.erode(image, structuring_element)

# Perform image dilation
opened_image = cv2.dilate(eroded_image, structuring_element)

# Display the original image and the opened image
stacked_results = np.hstack((image, opened_image)) 
cv2.imshow('Opened Image', stacked_results)

# Wait for key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_images/opened.jpg", stacked_results)