import cv2 as cv

cap = cv.VideoCapture("src/videos/videoplayback.mp4")
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.6, fy=0.6)
    cv.imshow("Video", frame)
    key = cv.waitKey(12)
    if key == 27:
        break