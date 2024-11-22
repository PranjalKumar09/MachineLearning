### Drawing Shapes, Text, and Image Manipulations with OpenCV

#### 1. **Drawing a Rectangle**
```python
left_top_point = (0, 0)
right_bottom_point = (250, 250)
color = (0, 250, 250)  # Cyan color in BGR format
thickness = 4

cv.rectangle(blank_image, left_top_point, right_bottom_point, color, thickness=thickness)
```
- **Thickness**: `4` for border thickness. Use `thickness=-1` to fill the rectangle.

#### 2. **Drawing a Line**
```python
right_bottom_point = (250, 250)
color = (0, 250, 250)

cv.line(blank_image, point1, point2, color, thickness=thickness)
```

#### 3. **Rectangle with Dynamic Size (Quarter Fill)**
```python
right_bottom_point = (image.shape[1]//2, image.shape[0]//2)  # Quarter size based on image dimensions
cv.rectangle(blank_image, left_top_point, right_bottom_point, color, thickness=2)
```
- The rectangle will occupy the top-left quarter of the image.

#### 4. **Drawing a Circle**
```python
mid_point = (250, 250)
radius = 40
cv.circle(blank_image, mid_point, radius, color, thickness=thickness)
```

#### 5. **Adding Text to an Image**
```python
left_top_point = (0, 0)
point = (250, 250)
color = (0, 250, 250)
thickness = 4
font_scale = 1.0
font_style = 1

cv.putText(blank_image, "This is Text", point, font_style, font_scale, color, thickness)
```
- **Text**: Placed at `point` `(250, 250)`.
- **Font**: `cv2.FONT_HERSHEY_SIMPLEX` (or `1`).
- **Color**: Cyan.
- **Font Scale**: `1.0` (adjusts text size).
- **Thickness**: `4`.

#### 6. **Converting Image to Grayscale**
```python
gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
cv.imshow("Gray", gray)
cv.waitKey(0)
```
- **Grayscale**: Converts the image to grayscale, reducing the color channels to one.

#### 7. **Blurring the Image**
```python
blur = cv.GaussianBlur(image, (3, 3), cv.BORDER_DEFAULT)
cv.imshow("Blurred Image", blur)
```
- **Gaussian Blur**: Reduces noise and details using a Gaussian kernel of size `(3, 3)`.

#### 8. **Edge Detection Using Canny**
```python
canny = cv.Canny(image, 125, 175)
cv.imshow("Canny Edge Detection", canny)
```
- **Canny Edge Detection**: Detects edges using the thresholds `125` and `175` for edge detection.
  
#### 9. **Edge Detection with Pre-blurred Image**
```python
canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny Edge Detection2", canny)
```
- **Canny After Blurring**: Edge detection on a blurred image to reduce noise.

#### 10. **Resizing an Image**
```python
destination_size = (500, 500)
resized = cv.resize(image, destination_size)
```
- **Resize**: Changes the image size to `(500, 500)`.
- Its like `cv.resize(image, destination_size, Interpolation method)`

##### **Interpolation Methods for Resizing**
- **`cv.INTER_LINEAR`**: Default interpolation, good for enlarging.
- **`cv.INTER_NEAREST`**: Fast but lower quality, uses nearest pixel.
- **`cv.INTER_CUBIC`**: Slower but better quality for enlarging images (uses 4x4 pixel interpolation).
- **`cv.INTER_LANCZOS4`**: High-quality interpolation for downsampling.

#### 11. **Cropping an Image**
```python
cropped = resized[50:450, 100:400]
cv.imshow("Cropped Image", cropped)
```
- **Cropping**: [row_range, column_range]
