import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv.imread("Image/colourful_butterfly.jpg")

# Create a circular mask
mask = np.zeros(img.shape[:2], dtype="uint8")
center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
radius = 100
cv.circle(mask, (center_x, center_y), radius, 255, -1)

# Apply the mask
masked_img = cv.bitwise_and(img, img, mask=mask)

# Grayscale Histogram for Masked Image
gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram (Masked Image)")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.plot(gray_hist, color='gray')
plt.xlim([0, 256])

# Color Histograms for Masked Image
colors = ('b', 'g', 'r')
plt.figure()
plt.title("Color Histogram (Masked Image)")
plt.xlabel("Bins")
plt.ylabel("Frequency")

for i, color in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])  # Calculate histogram
    plt.plot(hist, color=color)  # Plot histogram
    plt.xlim([0, 256])  # Set x-axis limits

# Show results
cv.imshow('Original Image', img)
cv.imshow('Mask', mask)
cv.imshow('Masked Image', masked_img)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
