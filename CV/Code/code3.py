import cv2 as cv
import numpy as np

image =  cv.imread("Image/cnn.png")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)


canny = cv.Canny(image, 125, 175)
cv.imshow('Canny', canny)

contours, hierachies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found')
""" 
CHAIN_APPROX_NONE is default type and pick all contours
CHAIN_APPROX_SIMPLE takes as contour of line as 2 point 
"""

# we can even draw the contours
blank = np.zeros(image.shape, np.uint8)
cv.drawContours(blank, contours, -1, (0, 255, 0), thickness=1)
cv.imshow('Contours', blank)





threshold = 125
maxval = 255
ret, thresh = cv.threshold(gray, threshold, maxval, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)


contours, hierachies = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found')

# very kess cotours are found in binary threshoold image


# it is recommended to use canny first


cv.waitKey(0)