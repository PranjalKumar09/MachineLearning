### Notes for `FaceDetector` Class & Face Mesh Detection with OpenCV and Mediapipe:

1. **Imports**:
   - `cv2`: OpenCV for image processing and capturing video frames.
   - `mediapipe`: Library for face mesh detection and drawing utilities.
   - `time`: Used to calculate FPS.

2. **Class: `FaceDetector`**:
   - **Constructor (`__init__`)**:
     - **Parameters**:
       - `staticMode`: Whether to use the static image mode (default `False`).
       - `maxFaces`: Maximum number of faces to detect (default `2`).
       - `minDetectionCon`: Minimum detection confidence (default `0.5`).
       - `minTrackCon`: Minimum tracking confidence (default `0.5`).
     - Initializes Mediapipe's `FaceMesh` with the given parameters.
     - Creates drawing specs for visualizing landmarks (circle radius and thickness for lines).

   - **Method: `findFaceMesh`**:
     - **Parameters**:
       - `img`: The image to process.
       - `draw`: Whether to draw landmarks on the image (default `True`).
     - **Functionality**:
       - Converts the image to RGB for processing with Mediapipe.
       - Uses `faceMesh.process(imgRGB)` to detect landmarks.
       - If faces are detected, draws landmarks on the image (if `draw=True`).
       - Appends each detected face's landmarks to a `faces` list.
       - Returns the processed image and list of face landmarks (`id`, `(x, y)`).

3. **`main` Function**:
   - **Webcam Setup**:
     - Uses `cv2.VideoCapture(0)` to capture video from the webcam.
   - **Face Detection Loop**:
     - Reads frames from the webcam and calls `findFaceMesh` to detect and draw landmarks.
     - Calculates FPS using the `time` module to track the time between frames.
     - Displays the processed image with FPS.
   - **Exit**:
     - Exits the loop when the 'q' key is pressed.

4. **Execution Block (`if __name__ == "__main__":`)**:
   - Ensures that `main()` is only executed when the script is run directly (not imported).

5. **Key Concepts**:
   - **Face Mesh**: Detects 468 facial landmarks on the face (e.g., eyes, nose, mouth).
   - **FPS Calculation**: Tracks and displays frames per second to monitor the processing speed.
   - **Drawing**: Visualizes detected landmarks on the face in the webcam feed.

6. **Notes**:
   - The face landmarks are drawn using Mediapipe’s `drawing_utils` which allows for customizable drawing of the mesh.
   - The `findFaceMesh` method processes the frame and appends each face's landmarks (positioned in image coordinates) to a list.
   - Press 'q' to exit the webcam feed.

---

This concise breakdown covers the essentials of the code for face detection and visualization using Mediapipe's `FaceMesh` model with OpenCV.


git log --oneline
b6d53da

