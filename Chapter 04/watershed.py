import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("input_images/4_SearchImage.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel_size = 5
kernel = np.ones((kernel_size,kernel_size),np.uint8)

# Find area which is surely background
sure_bg = cv2.dilate(thresh,kernel,iterations=1)

# Find are which is surely foreground
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
_, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)

# Find the region which is neither surely foreground nor surely background
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(image,markers)
image[markers == -1] = [0,0,0]



# The next 3 steps are needed only for better visibilty in publishing. 
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
image = cv2.erode(image, structuring_element)
image = cv2.erode(image, structuring_element)

cv2.imshow('watershed', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_images/watershed.jpg",image)