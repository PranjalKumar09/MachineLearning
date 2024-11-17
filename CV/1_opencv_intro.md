
### 1. **Installing Libraries**

- **OpenCV with Contrib Modules**:
  ```bash
  pip install opencv-contrib-python
  ```
  - Includes extra OpenCV functionalities.

- **Custom Helper Module**:
  ```bash
  pip install caer
  ```
  - Install `caer` for custom modules and utilities.

---

### 2. **Basic OpenCV Operations**

- **Reading and Showing Images**:
  ```python
  img = cv.imread("path")          # Read image from path
  cv.imshow('Window_name', img)    # Display image in a window
  cv.waitKey(0)                    # Wait indefinitely for key press
  ```

- **Handling Large Images**:
  - OpenCV doesn’t handle large images off-screen by default. Rescaling or window size adjustment may be necessary.

---

### 3. **Video Capture**

- **Capture Video**:
  ```python
  capture = cv.VideoCapture("path")  # Capture video from file
  capture = cv.VideoCapture(0)       # Capture video from default camera
  capture = cv.VideoCapture(1)       # Capture from 1st external camera
  capture = cv.VideoCapture(2)       # Capture from 2nd external camera
  ```

- **Read and Display Video Frame-by-Frame**:
  ```python
  while True:
      ret, frame = capture.read()        # Read a frame from video
      cv.imshow('Video', frame)          # Show the frame in a window

      if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
          break

  capture.release()                       # Release video capture
  cv.destroyAllWindows()                  # Close all OpenCV windows
  ```

- **Fixing `cv.imshow()` Error**:  
  If `frame` is `None` (e.g., when video finishes), you'll get an error:
  ```python
  cv2.error: OpenCV(4.10.0) /io/opencv/modules/highgui/src/window.cpp:973: error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'imshow'
  ```

---

### 4. **Rescaling Images/Videos**

- **Rescale Image/Frame**:
  ```python
  def rescaleFrame(frame, scale=0.75):
      height = int(frame.shape[0] * scale)
      width = int(frame.shape[1] * scale)
      dimensions = (width, height)
      return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
  ```

  - Use `rescaleFrame` to resize video frames for display.

- **Change Resolution for Live Video**:
  ```python
  def changeRes(width, height):
      capture.set(3, width)    # Set width
      capture.set(4, height)   # Set height
  ```

---

### 5. **Creating and Modifying Blank Images**

- **Create Blank Image**:
  ```python
  blank_image = np.zeros((500, 500), dtype='uint8')
  cv.imshow('blank_image', blank_image)
  cv.waitKey(0)
  ```

- **Create Blank Colored Image**:
  ```python
  blank_image = np.zeros((500, 500, 3), dtype='uint8')
  blank_image[:] = 0, 255, 0  # Set the color to green
  cv.imshow('green_image', blank_image)
  ```

- **Draw a Rectangle**:
  ```python
  blank_image[200:300, 323:500] = 0, 0, 255  # Set a red rectangle on green background
  cv.imshow('green_image_with_red_rectangle', blank_image)
  cv.waitKey(0)
  ```
