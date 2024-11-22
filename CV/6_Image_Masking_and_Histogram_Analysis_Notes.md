### **Learning Notes: Image Masking and Histograms in OpenCV**



#### **1. Applying a Mask**
  ```python
  masked_img = cv.bitwise_and(img, img, mask=mask)
  cv.imshow('Masked Image', masked_img)
  ```

---

#### **2. Calculating Histograms**
##### **Grayscale Histogram**
- Convert the image to grayscale:
  ```python
  gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
  ```
- Calculate the histogram for the masked grayscale image:
  ```python
  gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
  ```
- 256 denotes the on no of bins
- Plot the histogram:
  ```python
  plt.figure()
  plt.title("Grayscale Histogram (Masked Image)")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.plot(gray_hist, color='gray')
  plt.xlim([0, 256])
  ```

##### **Color Histograms (B, G, R)**
- For each color channel (blue, green, red), calculate and plot histograms:
  ```python
  colors = ('b', 'g', 'r')
  for i, color in enumerate(colors):
      hist = cv.calcHist([img], [i], mask, [256], [0, 256])  # Masked histogram
      plt.plot(hist, color=color)  # Plot with respective color
      plt.xlim([0, 256])
  ```
- Show the plot:
  ```python
  plt.show()
  ```

---


#### **Key Learning Points**

1. **Histograms**:
   - Show pixel intensity distribution.
   - Can be calculated for grayscale and individual color channels.

2. **Applications**:
   - Analyze color/intensity distribution in a specific region of an image.
   - Useful in computer vision tasks like segmentation, feature extraction, and filtering.

---

These notes provide a structured understanding of the code and its concepts for effective learning.