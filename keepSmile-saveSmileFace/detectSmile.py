import cv2 as cv
import time
import random
import string

face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_smile.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while(True):
    ret, frame = cap.read()

    if not ret:
        print("无法获取画面帧")
        break

    faces = face_cascade.detectMultiScale(frame,1.3,2)
    img = frame

    flag = False
    for (x, y, w, h) in faces:
        face_area = img[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(face_area,scaleFactor = 1.16,minNeighbors = 65,minSize = (25,25) , flags = cv.CASCADE_SCALE_IMAGE)
        if len(smiles):
            if len(img[y - 10:y + h + 20, x - 10:x + w + 10]):
                grayImag = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                canny = cv.Canny(grayImag, 200, 200)
                value = canny.var()

                lapla = cv.Laplacian(grayImag, cv.CV_8U)
                imageVar = lapla.var()
                if imageVar >= 100:
                    cv.imwrite(f"face_output/{round(time.time())}{''.join(random.sample(string.ascii_letters + string.digits, 6))}.png", img[y - 10:y + h + 10, x - 10:x + w + 10])
                    flag = True



    cv.imshow('Keep Your Smile', img)
    if flag:
        time.sleep(2)

    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()

cv.destroyAllWindows()


