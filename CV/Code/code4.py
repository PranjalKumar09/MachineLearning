import cv2 as cv

import numpy as np

img = cv.imread("Image/cnn.png")

# splitting the image

b,g,r  = cv.split(img)
blank = np.zeros(img.shape[:2], dtype='uint8')

blue  = cv.merge([b, blank, blank])
green = cv.merge([blank,g, blank])
red  = cv.merge([blank, blank, r])



cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)
print(blue.shape)
print(green.shape)
print(red.shape)



print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)
""" 
(551, 940, 3)
(551, 940)
(551, 940)
(551, 940)
"""

merged = cv.merge([b,g,r])
cv.imshow('Merged',merged)
""" 
we get original image

"""


cv.waitKey(0)
import cv2 as cv

import numpy as np

img = cv.imread("Image/cnn.png")

# splitting the image

b,g,r  = cv.split(img)
blank = np.zeros(img.shape[:2], dtype='uint8')

blue  = cv.merge([b, blank, blank])
green = cv.merge([blank,g, blank])
red  = cv.merge([blank, blank, r])



cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)
print(blue.shape)
print(green.shape)
print(red.shape)



print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)
""" 
(551, 940, 3)
(551, 940)
(551, 940)
(551, 940)
"""

merged = cv.merge([b,g,r])
cv.imshow('Merged',merged)
""" 
we get original image

"""


cv.waitKey(0)
