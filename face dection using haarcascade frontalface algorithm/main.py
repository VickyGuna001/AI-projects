import cv2
alg = r"D:\project\face dection using haarcascade frontalface algorithm\haarcascade_frontalface_default.xml" #importing algorithms
haar_cascade = cv2.CascadeClassifier(alg) #loading the algorithm
cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(gray_img, 1.3, 4) # detect the multiface in frame
    for (x, y, w, h) in face: #get coordinates from the faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
