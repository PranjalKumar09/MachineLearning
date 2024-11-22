### Notes: Face Detection Using Mediapipe and OpenCV

##### **1 Constructor: `__init__(self, minDetectionCon=0.5)`**
- **Parameters**:
  - `minDetectionCon` (float): Minimum confidence for detecting a face.
- **Initialization**:
  - `mpFaceDetection`: Mediapipe face detection module.
  - `mpDraw`: Mediapipe utility for drawing annotations.
  - `faceDetection`: Mediapipe's `FaceDetection` object configured with confidence threshold.

---

##### **2.1 Method: `findFaces(self, img, draw=True)`**
- **Purpose**: Detect faces and optionally draw bounding boxes.
- **Process**:
  1. Convert `img` to RGB (as Mediapipe requires RGB format).
  2. Use `self.faceDetection.process(imgRGB)` to detect faces.
  3. Extract bounding boxes using:
     - `detection.location_data.relative_bounding_box` (normalized coordinates).
  4. Convert normalized coordinates to pixel values using `img.shape`:
     - `x`, `y`: Top-left corner of the bounding box.
     - `w`, `h`: Width and height of the bounding box.
  5. Append bounding box `[id, x, y, w, h]` to `bboxes` list.
  6. If `draw=True`, call `fancyDraw()` to draw bounding boxes.
  7. Optionally, display detection confidence on the image.
- **Returns**:
  - `img`: Image with bounding boxes (if `draw=True`).
  - `bboxes`: List of bounding boxes.

---

##### **2.2 Method: `fancyDraw(self, img, bbox, l=30, t=5, rt=1)`**
- **Purpose**: Draw a "fancy" bounding box with styled corners.
- **Parameters**:
  - `bbox`: Bounding box `[x, y, w, h]`.
  - `l`: Length of corner lines.
  - `t`: Thickness of corner lines.
  - `rt`: Radius for the main rectangle.
- **Steps**:
  1. Extract bounding box coordinates: `x`, `y`, `w`, `h`.
  2. Draw:
     - Main rectangle using `cv2.rectangle`.
     - Corner lines for enhanced visualization using `cv2.line`.
- **Returns**: Modified `img`.

---

#### **3. Main Function: `main()`**
- **Purpose**: Capture live video feed, detect faces, and display results with FPS.
- **Steps**:
  1. Initialize:
     - `cv2.VideoCapture`: For webcam feed.
     - `FaceDetector`: Instance of `FaceDetector` class.
     - `pTime`: Used to calculate FPS.
  2. Video loop:
     - Capture each frame using `cap.read()`.
     - Use `findFaces()` to detect faces and draw bounding boxes.
     - Calculate FPS:
       - `fps = 1 / (cTime - pTime)` where `cTime` is the current time.
       - Update `pTime` after calculating FPS.
     - Display FPS on the top-left corner using `cv2.putText`.
     - Show the processed image in a window titled "Face Detection".
  3. Exit loop:
     - Exit the loop when 'q' is pressed.
     - Release the camera and destroy all OpenCV windows.

---

#### **4. Notes on Mediapipe Detection**
- **Bounding Box**:
  - Normalized format (range 0 to 1) relative to the image size.
  - Attributes:
    - `xmin`, `ymin`: Top-left corner.
    - `width`, `height`: Width and height of the box.
- **Accessing Detection Info**:
  - `results.detections`: List of detected faces.
  - Confidence score: `detection.score[0]` (float between 0 and 1).

---

#### **5. Example Output**
- **FPS Display**: Real-time frame rate is displayed on the top-left corner.
- **Bounding Boxes**:
  - Regular bounding boxes with enhanced visualization via `fancyDraw`.
  - Confidence scores displayed above the bounding box.
