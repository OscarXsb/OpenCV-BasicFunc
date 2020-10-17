
import cv2

img = cv2.imread('import.png')

if img is None:
    print("未能读入图像，请检查图像文件路径是否正确")

cv2.imshow("Display windows (Press S to save)", img)

k = cv2.waitKey(0)

if k == ord("s"):
    cv2.imwrite("saved_img.png", img)

cv2.destroyAllWindows()
