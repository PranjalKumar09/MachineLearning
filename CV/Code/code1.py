import cv2 as cv

""" img = cv.imread("Image/cnn.png")
cv.imshow('Cat', img)
cv.waitKey(0)
"""
def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    """ only works for live video """
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture("Video/animal_windows.mp4")

while True:
    ret, frame = capture.read()
    
    resize = rescaleFrame(frame)
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()


