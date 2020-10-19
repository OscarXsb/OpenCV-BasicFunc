# OpenCV-BasicFunc
![](https://img.shields.io/badge/Python-v3.7.6-2EA44F.svg)  ![](https://img.shields.io/badge/OpenCV_for_Python-v4.4.0-297DDC.svg)  ![](https://img.shields.io/badge/PyTorch-v10.2-EE4C2C.svg)

这是一个学习如何最基础的使用 OpenCV-Python 与 Yolo 的开源源代码仓库。

### 开发环境
---
​    在本文中，所有的例程全部建立在 Python v3.7.6 的基础上，OpenCV - python 的版本是 v4.4.0，对于开发环境的配置，对于yolo的使用和开发，这里的Pytorch版本是v10.2，建议先安装 Anaconda 的Python 集成库，这样会方便新的包的安装和编译。
​    首先需要安装 pip 3，如果你在国内，建议将下载源切换为清华或者阿里的下载源，使用 **pip3 install opencv-python** 的命令安装 OpenCV python 依赖库，Pythorch 按照[官网](https://pytorch.org/)的步骤进行 ，配置好环境，就可以进行程序的编写了。

### 程序设计和代码学习
---
  #### OpenCV基础

​    首先，简单对 OpenCV 的基础运用进行了解。

>##### 摄像头的捕捉

​	因为OpenCV通常用于人脸识别等多个需要捕捉实时视频画面的应用，所以先看下如何打开摄像头。 源代码文件为 **Camera-10-11.py** 

代码如下：

```python
import cv2 #引入OpenCV-python 依赖库

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #设置 cap 为摄像头画面

cap.open(0) # 打开摄像头

while cap.isOpened():
	flag, frame = cap.read() #检测是否正常打开

	cv2.imshow('my_window',frame) #显示摄像头这一帧的画面

	key_pressed = cv2.waitKey(60) #等待 60 秒按键，这个数字与摄像头的获取有关
	print('The key you pressed is: ',key_pressed) # 检测按下的键
	if key_pressed == 27: #若按键为 Esc 则退出
		break

cap.release() #释放资源

cv2.destroyAllWindows() #销毁窗口
```



> ##### OpenCV 边缘化处理

源代码文件为 **Edge-detection-10-11.py** ，代码与注释如下：

```python
import cv2

import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 0代表默认摄像头

cap.open(0)

while cap.isOpened():
	flag, frame = cap.read()
	#返回一帧画面
	if not flag:
		break

	key_pressed = cv2.waitKey(60)
	print('The key you pressed is: ', key_pressed)

	frame = cv2.Canny(frame,100,200) #边缘检测

	#摞成三通道图像
	frame = np.dstack((frame,frame,frame))

	cv2.imshow('my_window',frame)
	if key_pressed == 27:
		break

cap.release()

cv2.destroyAllWindows()
```



> ##### OpenCV 人脸、眼睛、微笑检测

源代码文件为 **Smile-detection-10-11.py**，代码与注释如下：

```python
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #引入预置的人脸模型
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml') #引入预置的人眼模型
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml') #引入预置的微笑模型

cap = cv2.VideoCapture(0)
# 0代表默认摄像头

while(True):
    ret,frame = cap.read()
    faces = face_cascade.detectMultiScale(frame,1.3,2) #设置检测人脸的参数
    img = frame
    for(x,y,w,h) in faces: #检测多个人脸
        img = cv2.rectangle(img,(x,y),(x+w,x+h),(255,0,0),2) #用绿色方框圈出人脸所在位置，注意：（255，0，0）通道为 GBR

        face_area = img[y:y+h,x:x+w] #标志人脸位置

        eyes = eye_cascade.detectMultiScale(face_area,1.1,10) #设置检测眼的参数
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1) #用蓝色方框圈出人眼的位置

        smiles = smile_cascade.detectMultiScale(face_area,scaleFactor = 1.16,minNeighbors = 65,minSize = (25,25) , flags = cv2.CASCADE_SCALE_IMAGE) #设置检测微笑的参数
        for(ex,ey,ew,eh) in smiles:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,0,255),1) #标识位置
            cv2.putText(img,'Smile',(x,y-7),3,1.2,(0,0,255),2,cv2.LINE_AA) #放置文字

    cv2.imshow('frame2',img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
```

>##### 引入图片或视频

在OpenCV的处理过程中，引入本地已有的图片或视频进行处理也是必不可少的，这里只介绍视频的引入（图片的引入在 **Import-image/Import-image-10-14.py**，内含图片的导入与保存），源代码文件在 **Progress-video/Playing-videos-10-14.py** ,这里代码中引入的视频文件名为 test.mp4，需要自行准备，源代码如下：

```python
import cv2

cap = cv2.VideoCapture("test.mp4") #请注意，这里处理的视频是本地的视频，若为数字则为从个摄像头输入

if not cap.isOpened(): #校验是否成功导入
    print("无法打开视频")
    exit()

print('WIDTH', cap.get(3)) 
print('HEIGHT', cap.get(4)) #输出视频的高和宽

while True:
    ret, frame = cap.read() #逐帧读取视频

    if not ret: #校验是否成功读取帧
        print("无法获取画面帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #进行灰度处理，这样的处理会方便在以后的识别处理中速度更快

    cv2.imshow('frame_window', gray)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

以上就是OpenCV的基础，暂时告一段落

#### Yolo目标检测

​        YOLO（You Only Look Once）是一种基于深度神经网络的对象识别和定位算法，其最大的特点是运行速度很快，可以用于实时系统。Yolo 的工作原理不再进行分析，具体请见  [Github Yolo v5](https://github.com/ultralytics/yolov5)，首先需要下载 Yolo 的运行库 ，在这里进行克隆 [Github Yolo v5](https://github.com/ultralytics/yolov5)，官方已有训练好的模型，国内可在此下载，下载好后放置在 yolov5 库的 weights 文件夹下。

> ##### Yolo初体验

​       若要体验 Yolo v5 的目标检测结果，可在yolov5库下运行名为 detect.py 文件，后面的参数有两项必传，source 和 weights ，source 代表需进行识别的视频或图片文件，若为本地文件则输入文件路径，若选择默认摄像头，则传入 0 ，weights 代表具体引用的模型文件，上文提供下载的模型其文件权重不同，即识别准确率不同，当然，识别的速度也和模型文件有很大关系，若想调用摄像头，准确度最高，则在 cmd 中输入 **python detect.py --source 0 --weights weights/yolov5x.pt** ,即可进行识别，此处 **weights/yolov5x.pt** 为相对路径，x代表最大，最高，根据各个模型文件也可推断出准取度。

识别结果如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_1.png)







​    

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_2.jpg)

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_3.jpg)

