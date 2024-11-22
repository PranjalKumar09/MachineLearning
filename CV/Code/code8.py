"""
Face Detection -> 
    haarcasacade : classifer
    opencv have lot of classfier in their website like for eye, face etc
    
    -> https://github.com/opencv/opencv/tree/4.x/data/haarcascades
"""


# Import necessary libraries
import cv2 as cv

img = cv.imread("Image/family1.jpg")
# cv.imshow("Person",img)
def rescaleFrame(frame, scale=0.25):
      height = int(frame.shape[0] * scale)
      width = int(frame.shape[1] * scale)
      dimensions = (width, height)
      return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
img = rescaleFrame(img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Person",gray)

haar_cascade = cv.CascadeClassifier('CV/Code/haar_face.xml')

faces_rect =  haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=6)

print(f'Number of faces found: %d' %len(faces_rect))


for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
cv.imshow("Detected Faces", img)

""" 
in multiple person page, its show incorect because its highly sesitivity to flucations we can change it by changing minNeighbors in faces_rect

minimizing it , will more flucation of face
howerver increasing it can reduce image in real image
"""




cv.waitKey(0)
