### Notes for Pose Detection Script

#### **Imports and Initialization**
**Class Initialization (`poseDetector`)**:
   - **Parameters**:
     - `mode`: If `True`, processes each frame as a static image (useful for single images).
     - `model_complexity`: Determines the complexity of the landmark detection model (`1` for default, can be `0` or `2`).
     - `smooth_landmarks`: Enables smoothing for landmark movement in videos.
     - `detection`: Minimum confidence for detecting poses.
     - `trackCon`: Minimum confidence for tracking poses across frames.

---

#### **Key Methods**
1. **`findPose(img, draw=True)`**:
   - **Process**: Converts the image to RGB, processes it to detect pose landmarks.
   - **Output**: The modified image with landmarks drawn (if `draw=True`).

2. **`findPosition(img, draw=True)`**:
   - **Input**: `img` (image/frame), `draw` (flag to draw individual landmarks).
   - **Process**:
     - Iterates through detected landmarks.
     - Converts normalized coordinates to pixel coordinates.
     - Appends landmarks to a list `lmList` as `[id, x, y]`.
   - **Output**: List of landmarks (`lmList`).
   - **Additional Feature**: Draws circles on specific landmarks if `draw=True`.

---

#### **Main Script**
1. **Video Capture**:
   - Uses OpenCV's `cv2.VideoCapture(0)` to capture video from the webcam.

2. **Pose Detection Loop**:
   - Reads each frame, passes it to `findPose` for pose landmarks, and optionally gets their positions via `findPosition`.
   - Draws a specific landmark (e.g., ID `23`) with a large circle for clarity.

3. **FPS Calculation**:
   - Computes FPS to monitor real-time performance using timestamps.

4. **Visualization**:
   - Draws the FPS on the image.
   - Resizes the frame to `800x600` for better display and opens it in a window.

5. **Exit Condition**:
   - Press the `q` key to exit the loop and release resources.

---

#### **Key Landmarks**
Mediapipe provides landmarks indexed numerically (e.g., `23` for the left hip):
- To visualize or debug specific landmarks:
  - Use `lmList` and the corresponding landmark ID.
  - Draw circles, print coordinates, or use for further analysis.
