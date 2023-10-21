import cv2
import numpy as np

# Read the input image
image = cv2.imread("input_images/4_ThresholdingImage.jpg")  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    exit()

# Define the kernel for de-noising convolution
kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

# Perform image convolution
convolved_image = cv2.filter2D(image, -1, kernel)


stacked_results = np.hstack((image, convolved_image)) 
# Display the original image and the convolved output image
cv2.imshow('Convolution', stacked_results)

# Wait for key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_images/convolution.jpg", stacked_results)