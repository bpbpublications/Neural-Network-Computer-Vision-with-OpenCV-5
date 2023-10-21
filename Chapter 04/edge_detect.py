import cv2
import numpy as np
import sys

# Read the input image
image = cv2.imread("input_images/4_ThresholdingImage.jpg")  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    sys.exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply different edge detection algorithms
canny_edges = cv2.Canny(gray, 100, 200)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

# Create a window to display the images
cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)

# Display the original image and edge detection results side by side
stacked_results = np.hstack((canny_edges, sobel_x, sobel_y, laplacian_edges))
cv2.imshow('Edge Detection', stacked_results)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output_images/edge_detect.jpg", stacked_results)