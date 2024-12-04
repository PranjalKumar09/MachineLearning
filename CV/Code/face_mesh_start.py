import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
pTime = time.time()

# Initialize Mediapipe Face Mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# Access FACE_CONNECTIONS properly
FACE_CONNECTIONS = mpFaceMesh.FACEMESH_TESSELATION

while True:
    # Read the video frame
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face, FACE_CONNECTIONS, drawSpec, drawSpec)
            
            for id, ln in enumerate(face.landmark):
                # Get the normalized coordinates (x, y, z) for each landmark
                h, w, c = img.shape
                x, y, z = int(ln.x * w), int(ln.y * h), ln.z
            
                
                
                # x, y = ln.x * img.shape[1], ln.y * img.shape[0]
                # cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
                # cv2.putText(img, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# there are around more than 460 landmarks