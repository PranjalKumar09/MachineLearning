import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)

pTime = time.time()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Get frame dimensions
    h, w, c = img.shape

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Draw landmarks and connections on the original frame
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Iterate through landmarks for additional processing or debugging
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # Convert normalized coordinates to pixel values
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    img = cv2.resize(img, (800, 600))  # Resize to 800x600 for better visibility

    cv2.imshow("Pose Estimation", img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
