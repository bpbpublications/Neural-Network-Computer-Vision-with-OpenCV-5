import cv2
import numpy as np
import sys

# Read the input image
image = cv2.imread("input_images/4_ThresholdingImage.jpg", 0)  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    sys.exit()

# Get the thresholding level from command line argument
threshold_level = int(sys.argv[1])

# Apply different thresholding algorithms
ret, thresh_binary = cv2.threshold(image, threshold_level, 255, cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(image, threshold_level, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(image, threshold_level, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(image, threshold_level, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(image, threshold_level, 255, cv2.THRESH_TOZERO_INV)
ret, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

stacked_results = np.hstack((thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv, thresh_otsu))
# Create a window to display the thresholded images
cv2.namedWindow('Thresholding', cv2.WINDOW_NORMAL)
cv2.imshow('Thresholding', stacked_results)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output_images/thresholding.jpg", stacked_results)