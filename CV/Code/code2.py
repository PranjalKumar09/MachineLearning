import cv2 as cv
import numpy as np

image =  cv.imread("Image/cnn.png")

# resized

destination_size = (500,500)
resized = cv.resize(image, (500,500))
# cv.imshow("resized image", resized)

# also explain different interpolaiton in short


# Cropping

cropped = resized[50:450, 100:400]
cv.imshow("cropped image", cropped)


cv.waitKey(0)