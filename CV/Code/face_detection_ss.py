import cv2
import mediapipe as mp
import time

cap  = cv2.VideoCapture(0)
pTime = time.time()

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.25)




while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    # print(results) # <class 'mediapipe.python.solution_base.SolutionOutputs'>
    if results.detections: 
        for id, detection in enumerate(results.detections):
            # print(id , detection)# 0 label_id: 0
            # print(detection.score)# score: 0.955596566
            
            
            # print(detection.location_data.relative_bounding_box)
            """ [0.9555965662002563]
            xmin: 0.338085651
            ymin: 0.323026121
            width: 0.308925807
            height: 0.411878943 """


            # mpDraw.draw_detection(img, detection)
            ih , iw , ic = img.shape # hight, width, channels
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(img, bbox, (255,0,255), 2)
            
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
            
            

          
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,250, 0), 2)
    
    cv2.imshow("Capture" , img)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
# now bounding box by them is very unclear becuase of unaccurate box