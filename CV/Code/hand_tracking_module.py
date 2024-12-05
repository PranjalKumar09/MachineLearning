import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionConf=0.5,trackCon=0.5 ) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackCon = trackCon
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()  
        self.mp_draw = mp.solutions.drawing_utils
        
        self.pTime = 0
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hands
        self.results = self.hands.process(imgRGB)

        # Draw hand landmarks on the original BGR image if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks: # running through each image
                
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS) 
        return img
        
    def findPostion(self, img, handNo=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                x, y, z = ln.x * img.shape[1], ln.y * img.shape[0], ln.z  # converting the ratio to actual pixel
                lmList.append([id, int(x), int(y)])  # append the landmark to the list
                # if (id==0): # for palm
                #     cv2.circle(img, (int(x), int(y)), 15, (255, 0, 255), -1) # drawing circle on each landmark
                if draw:
                    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)  # drawing circle on each landmark
                    
        return lmList
        
        
    









# Initialize video capture and MediaPipe Hands
# cap = cv2.VideoCapture(0)










def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, img = cap.read() 
        img = detector.findHands(img)
        lmList = detector.findPostion(img, handNo=0)
        
        if len(lmList)!=0:
            print(lmList[0]) # palm
        
        # now lets show frame also
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # put FPS on frame


    
    # Display the video frame with landmarks
        cv2.imshow('Video', img)

    # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()