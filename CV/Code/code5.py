import cv2 as cv

import numpy as np

img = cv.imread("Image/cnn.png")
""" 
# blurring

# Averaging
average = cv.blur(img, (7,7))
cv.imshow('Average Blur', average)

# gaussian blur (it appear less bulr then averaging)
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow("GaussianBlur", gauss)

#  Median Blur (it appear less blur then above)
median = cv.medianBlur(img,3)
cv.imshow("Median Blur", median)

# Bilateral (it used more, not tradtiontal)
bilaterial = cv.bilateralFilter(img,10, 35, 25)
cv.imshow("Bilateral Filter",bilaterial) """
""" 
blank = np.zeros((400,400), dtype = 'uint8')

rectangle = cv.rectangle(blank.copy(), (30, 30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)


    # also explain what will happend in .copy removed or what is that doing

# bit wise .append()
bitwise_and = cv.bitwise_and(rectangle ,circle) # intersecting region
cv.imshow('bitwise_and', bitwise_and)

bitwise_or = cv.bitwise_or(rectangle ,circle ) # non intersecting + intersectiong
cv.imshow('bitwise_or', bitwise_or)

bitwise_xor = cv.bitwise_xor(rectangle ,circle ) # non intersectin
cv.imshow('bitwise_xor', bitwise_xor)

bitwise_not = cv.bitwise_not(rectangle ) # invert  colours
cv.imshow('bitwise_not', bitwise_not) """

# masking 

blank = np.zeros(img.shape[:2] , dtype='uint8')
# cv.imshow('Blank image',blank)
""" 
mask = cv.circle(blank ,( img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked', masked)

 """


# we can even create any weird shape 

circle = cv.circle(blank.copy(),  ( img.shape[1]//2-100, img.shape[0]//2+123), 100, 255, -1)
cv.imshow('Mask_circle', circle)

rectangle = cv.rectangle(blank.copy(),(30,39), (380,380) ,225 , -1)
cv.imshow('Mask_rectangle', rectangle)
weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('weird_shape', weird_shape)

masked = cv.bitwise_and(img, img, mask = weird_shape)
cv.imshow('Masked Image (weird)', weird_shape)



cv.waitKey(0)
