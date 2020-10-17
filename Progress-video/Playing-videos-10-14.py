import cv2

cap = cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    print("无法打开视频")
    exit()

print('WIDTH', cap.get(3))
print('HEIGHT', cap.get(4))

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法获取画面帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame_window', gray)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
