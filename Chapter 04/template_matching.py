import cv2
import numpy as np

# Read the search image and the template image
search_image = cv2.imread('input_images/4_SearchImage.jpg', cv2.IMREAD_COLOR)  
template_image = cv2.imread('input_images/4_Template.jpg', cv2.IMREAD_COLOR)  

# Check if the images were successfully loaded
if search_image is None or template_image is None:
    print("Unable to load the images.")
    exit()

# Convert the images to grayscale
search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Set a threshold for the match score
threshold = 0.6

# Find the locations where the match score is above the threshold
locations = np.where(result >= threshold)

# Draw rectangles around the matched regions
for pt in zip(*locations[::-1]):
    bottom_right = (pt[0] + template_gray.shape[1], pt[1] + template_gray.shape[0])
    cv2.rectangle(search_image, pt, bottom_right, (0, 255, 0), 2)

# Display the search image with the matched regions
cv2.imshow('Template Matching Result', search_image)

# Wait for key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_images/template_matching.jpg", search_image)