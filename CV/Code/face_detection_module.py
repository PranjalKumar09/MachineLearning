import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        """
        Initialize the FaceDetector class.

        Parameters:
        - minDetectionCon (float): Minimum detection confidence threshold.
        """
        self.minDetectionCon = minDetectionCon

        # Initialize Mediapipe Face Detection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Detect faces in the given image and return bounding boxes.

        Parameters:
        - img (numpy.ndarray): Input image.
        - draw (bool): Whether to draw bounding boxes on the image.

        Returns:
        - img (numpy.ndarray): Image with bounding boxes (if draw=True).
        - bboxes (list): List of bounding boxes, each in [id, x, y, w, h] format.
        """
        bboxes = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # Extract bounding box and normalize coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, x, y, w, h])

                # Draw bounding box and detection score
                if draw:
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    self.fancyDraw(img, [x, y, w, h])
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return img, bboxes
    
    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        """
        Draw a fancy bounding box with thicker corners.

        Parameters:
        - img (numpy.ndarray): Input image.
        - bbox (list): Bounding box in [x, y, w, h] format.
        - l (int): Length of corner lines.
        - t (int): Thickness of the lines.
        - rt (int): Radius of corners.

        Returns:
        - img (numpy.ndarray): Image with fancy bounding box.
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw main rectangle
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), rt)

        # Top-left corner
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)

        # Top-right corner
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)

        # Bottom-left corner
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)

        # Bottom-right corner
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)

        return img




def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    pTime = 0  # Previous time for FPS calculation

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video.")
            break

        # Detect faces and retrieve bounding boxes
        img, bboxes = detector.findFaces(img,  )

        # Calculate FPS
        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # FPS formula
        pTime = cTime  # Update previous time

        # Display FPS in top-left corner
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow("Face Detection", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
