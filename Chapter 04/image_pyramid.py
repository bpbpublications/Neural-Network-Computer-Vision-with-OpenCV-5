import cv2
import sys

# Read the input image
image = cv2.imread("input_images/4_Clustering.jpg")  

# Check if the image was successfully loaded
if image is None:
    print("Unable to load the image.")
    sys.exit()

# Display the original image
cv2.imshow('Original Image', image)

# Generate and display the image pyramid
pyramid_image = image.copy()
pyramid_images = [pyramid_image]

while pyramid_image.shape[0] > 100 and pyramid_image.shape[1] > 100:  # Adjust the condition to control the pyramid size
    pyramid_image = cv2.pyrDown(pyramid_image)
    pyramid_images.append(pyramid_image)

for i, image_level in enumerate(pyramid_images):
    cv2.imshow(f'Pyramid Level {i}', image_level)

# Wait for key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
