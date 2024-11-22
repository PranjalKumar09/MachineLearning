### Hand Detection with MediaPipe



#### **Overview**
- **Framework**: MediaPipe (developed by Google)  
- **Features**: 
  - Palm detection for initial hand localization.
  - 21 distinct hand landmarks for detailed tracking and gesture analysis.

---

#### **Code Explanation**

##### **1. Using MediaPipe with OpenCV**
The following example showcases real-time hand detection using MediaPipe's `Hands` solution with OpenCV:

- **Key Steps**:
  1. Initialize the camera feed using OpenCV.
  2. Process frames with MediaPipe's Hand model.
  3. Visualize hand landmarks and connections on the captured frames.

```python
import cv2
import time
import mediapipe as mp

# Initialize video capture and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()  
mp_draw = mp.solutions.drawing_utils
pTime = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert the image to RGB as required by MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(imgRGB)

    # Draw detected landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate and display the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video stream
    cv2.imshow('Video', img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

##### **2. Encapsulating Functionality with a Class**

The `handDetector` class simplifies hand detection and landmark extraction, offering reusability and modularity.

- **Features**:
  1. **`findHands`**: Detect hands and optionally draw landmarks.
  2. **`findPosition`**: Retrieve coordinates of individual hand landmarks.

```python
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                x, y = int(ln.x * img.shape[1]), int(ln.y * img.shape[0])
                lmList.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
        return lmList
```

---

#### **3. Using the `HandDetector` Class**
The class is integrated into a main loop for real-time detection.

```python
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(lmList[0])  # Log the position of the palm (landmark ID 0)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the video feed
        cv2.imshow('Video', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

---

#### **Notes on Functionality**
1. **Palm and Landmarks**:
   - The **palm** corresponds to **landmark ID 0**.
   - Each hand has **21 landmarks** indexed sequentially.

2. **Landmark Coordinates**:
   - MediaPipe provides normalized coordinates (ratios relative to image size).
   - These are converted to pixel values by scaling with the frame's width and height.

3. **Visualization**:
   - **`mp_draw.draw_landmarks`**: Draws landmarks and connections on the frame.
   - Custom shapes (e.g., circles) can be drawn using OpenCV.

4. **Performance Metrics**:
   - **FPS Calculation**: Measures the detection and rendering speed for real-time performance assessment.

---
