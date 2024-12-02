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

    # Convert BGR image to RGB as required by MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    results = hands.process(imgRGB)

    # Draw hand landmarks on the original BGR image if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks: # running through each image
            for id, ln in enumerate(hand_landmarks.landmark):
                x, y, z = ln.x * img.shape[1], ln.y * img.shape[0], ln.z  # converting the ratio to actual pixel
                if (id==0): # for palm
                    cv2.circle(img, (int(x), int(y)), 15, (255, 0, 255), -1) # drawing circle on each landmark
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # then drawing each point
    # hand_landmarks  are points , mp_hands.HAND_CONNECTIONS connection between them
    # hand_landmarks gives 3 points in decimal (which is ratio to repective size(hight , width))
    

    
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
