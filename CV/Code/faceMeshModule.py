import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initialize the FaceDetector class with configuration options.
        
        Parameters:
        - staticMode (bool): Whether to run the model in static mode (default is False).
        - maxFaces (int): Maximum number of faces to detect (default is 2).
        - minDetectionCon (float): Minimum detection confidence threshold (default is 0.5).
        - minTrackCon (float): Minimum tracking confidence threshold (default is 0.5).
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        # Initialize Mediapipe FaceMesh and drawing utils
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Detect face landmarks in the given image and return the image with landmarks drawn (if draw=True).
        
        Parameters:
        - img (numpy.ndarray): Input image.
        - draw (bool): Whether to draw landmarks on the image.
        
        Returns:
        - img (numpy.ndarray): Image with landmarks drawn (if draw=True).
        - faces (list): List of detected faces' landmarks in the form [id, (x, y)].
        """
        faces = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB as required by Mediapipe
        self.results = self.faceMesh.process(imgRGB)  # Process the image
        
        if self.results.multi_face_landmarks:  # If faces are detected
            for id, face in enumerate(self.results.multi_face_landmarks):
                # Traverse through each landmark in the face
                if draw:
                    self.mpDraw.draw_landmarks(img, face, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                
                # Collect face landmarks and append (id, x, y) to faces list
                face_landmarks = []
                for lm in face.landmark:
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_landmarks.append((x, y))  # Add (x, y) to the list
                
                faces.append([id, face_landmarks])  # Append face id and landmarks to faces list
        
        return img, faces


def main():
    # Set up webcam and FaceDetector instance
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

    pTime = time.time()  # Start time for FPS calculation

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Process the image and detect faces
        img, faces = detector.findFaceMesh(img, draw=True)

        # Calculate FPS
        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # FPS formula
        pTime = cTime  # Update previous time

        # Display FPS on the image
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with landmarks and FPS
        cv2.imshow("Face Mesh", img)

        # Exit loop on pressing 'q'
        if cv2.waitKey(250) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


# Entry point for running the module as a standalone script
if __name__ == "__main__":
    main()
