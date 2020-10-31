# OpenCV-BasicFunc
![](https://img.shields.io/badge/Python-v3.7.6-2EA44F.svg)  ![](https://img.shields.io/badge/OpenCV_for_Python-v4.4.0-297DDC.svg)  ![](https://img.shields.io/badge/PyTorch-v10.2-EE4C2C.svg)

这是一个学习如何最基础的使用 OpenCV-Python 与 Yolo 的开源源代码仓库。

###### *\*注意：本说明文件与安装过程均在  **Windows10** 上进行！*

### 开发环境
---
​    在本文中，所有的例程全部建立在 Python v3.7.6 的基础上，OpenCV - python 的版本是 v4.4.0，对于开发环境的配置，对于yolo的使用和开发，这里的Pytorch版本是v10.2，建议先安装 Anaconda 的Python 集成库，这样会方便新的包的安装和编译。
​    首先需要安装 pip 3，如果你在国内，建议将下载源切换为清华或者阿里的下载源，使用 **pip3 install opencv-python** 的命令安装 OpenCV python 依赖库，Pythorch 按照[官网](https://pytorch.org/)的步骤进行 ，配置好环境，就可以进行程序的编写了。

### 程序设计和代码学习
---
  #### OpenCV基础

​    首先，简单对 OpenCV 的基础运用进行了解。

##### 摄像头的捕捉
---
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



##### OpenCV 边缘化处理
---
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



##### OpenCV 人脸、眼睛、微笑检测
---
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

##### 引入图片或视频
---
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

#### Yolo目标检测基础

​        YOLO（You Only Look Once）是一种基于深度神经网络的对象识别和定位算法，其最大的特点是运行速度很快，可以用于实时系统。Yolo 的工作原理不再进行分析，具体请见  [Github Yolo v5](https://github.com/ultralytics/yolov5)，首先需要下载 Yolo 的运行库 ，在这里进行克隆 [Github Yolo v5](https://github.com/ultralytics/yolov5)，官方已有训练好的模型，国内可在此[下载](https://cloud.189.cn/t/aQvMnuzYfQvq)，下载好后放置在 yolov5 库的 weights 文件夹下。

##### Yolo初体验
---
​       若要体验 Yolo v5 的目标检测结果，可在yolov5库下运行名为 detect.py 文件，后面的参数有两项必传，source 和 weights ，source 代表需进行识别的视频或图片文件，若为本地文件则输入文件路径，若选择默认摄像头，则传入 0 ，weights 代表具体引用的模型文件，上文提供下载的模型其文件权重不同，即识别准确率不同，当然，识别的速度也和模型文件有很大关系，若想调用摄像头，准确度最高，则在 cmd 中输入 **python detect.py --source 0 --weights weights/yolov5x.pt** ,即可进行识别，此处 **weights/yolov5x.pt** 为相对路径，x代表最大，最高，根据各个模型文件也可推断出准确度。

识别结果如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_1.png)







​    

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_2.jpg)

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/yolov5_res_3.jpg)

##### Yolo模型训练
---
对于Yolo v5模型训练，这里只做简要演示，不作详细说明。

要进行模型训练，首先需要持有数据集，如果你要构建自己的项目，当然可以自行进行图片的的采集以及需要检测物体的标注，要进行标注的步骤，可以采用 MIT 的 labelImg，安装方法不做说明，如果只是想进行练习，可以从[这里](https://public.roboflow.com/)下载数据集，示例训练的是口罩检测的[模型](https://public.roboflow.com/object-detection/mask-wearing)，国内环境有限，所以[这里](https://cloud.189.cn/t/mUFZfmBJn67n)提供下载好的口罩数据集

做好准备工作,进入正题:

首先需要准备数据文件,这里mask文件夹与yolov5处于同一文件夹下,data文件位于下载好的压缩包中,文件如下:



```yaml
train: ../mask/train/images #相对于train.py的路径
val: ../mask/valid/images

nc: 2 #类型的数量
names: ['mask', 'no-mask'] #这里分为戴口罩和未戴口罩的状态,分别 依次对应数字 0,1
```

然后需要准备好训练时需要读取的文件如下,存放在 **yolov5/models/ **目录下,这里我想把精度调高一些,所以采用yolov5x.yaml进行修改,在同级目录下命名为yolov5x_mask.yaml :



```yaml
# parameters
nc: 2 # number of classes 只需要更改这里的数量
depth_multiple: 1.33  # model depth multiple
width_multiple: 1.25  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

进入到下一步,训练:

在 **yolov5/train.py** 所在目录下启动cmd, 输入 **python train.py --img 640 --batch 2 --epochs 1000 --data ../mask/data.yaml --cfg models/yolov5x_mask.yaml --weights weights/yolov5x.pt** 的命令,其中 epochs 后对应的是训练时迭代的次数,一般情况下 300 就足够了,batch后填写一次喂到模型里的数据量,正常应为16,由于GPU屡次报 内存溢出 的错误,故改为 2,weights代表权重文件,这里指定的是精度最高的文件,若输入 ' ' 则为随机,回车开始,在训练过程中,为了使训练过程可视化,可以在yolov5目录下新建cmd窗口,输入 tensorboard --logdir runs/ ,然后根据提示在浏览器中输入 **localhost:提示的端口号**,即可实时刷新看到效果,训练过程如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/train_yolov5_3.png)

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/train_yolov5_2.png)

实时预览如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/train_yolov5_1.png)

训练后模型使用效果如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/train_yolov5_3.jpg)s

Yolov5暂时先告一段落，下面为大家简单叙述人脸识别的相关应用。

#### Face Recognition
---
这里用到的是Github比较火爆的工具包 - **[face_recognition](https://github.com/ageitgey/face_recognition)** ,官方给出的描述如下：

> *Recognize and manipulate faces from Python or from the command line with the world's simplest face recognition library.*
>
> *Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition built with deep learning. The model has an accuracy of 99.38% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.*
>
> *This also provides a simple `face_recognition` command line tool that lets you do face recognition on a folder of images from the command line!*

> *使用世界上最简单的面部识别库从Python或命令行识别和操作面部。*
> *使用[dlib](http://dlib.net/)的最先进的面部识别技术和深度学习技术构建而成。 该模型在[野生动物标签脸](http://vis-www.cs.umass.edu/lfw/)基准上的准确性为 99.38％。*
> *这也提供了一个简单的“ face_recognition”命令行工具，使您可以从命令行对图像文件夹进行人脸识别！*

###### **安装**

根据 Github 上官方文档说明：

- macOS or Linux (Windows not officially supported, but might work)

但是由于本文的环境配置在 Linux 系统上，故给出 Windows 的安装教程（安装过程有些繁琐，但本文将给出详细说明）

1.如果本机没有安装 Visual Studio, 安装[Visual Studio]( https://visualstudio.microsoft.com/zh-hans/)，我选择的是 Professional 2019 版本，下载后选择使用 C++的桌面开发进行安装，提示重新启动电脑，重启后继续以下步骤。

2.安装 [boost](https://www.boost.org/users/download/),选择Windows平台下的 ZIP 文件，下载后解压，速度较慢，完成后，在 boost 目录下打开 cmd，输入 bootstrap.bat 运行，看到该目录下生成 b2.exe ，即可在命令行里运行 b2.exe,安装过程如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/face_rec_install_1.png)

成功后提示如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/face_rec_install_2.png)

3.接下来安装 [CMake](https://cmake.org/download/),选择该项进行下载， 

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/face_rec_install_3.png)

安装过程中，选择 **Add CMake to the system PATH for all users**, 安装成功后，进入到下一步。

4.安装 [dlib](http://dlib.net/),单击左下角的 **Download dlib** ,解压后，在目录下打开cmd，输入 **python setup.py install** 进行安装，成功后提示如下：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/face_rec_install_4.png)

5.打开 cmd ，输入 **pip3 install face_recognition** ,使用pip进行安装，成功后提示 **Successfully installed face-recognition-X.X.X (版本号) face-recognition-models-X.X.X (版本号)**

附：考虑到有些小伙伴网络环境不畅，因此将 boost_1_74_0.zip , dlib-19.21.zip , cmake-3.19.0-rc2-win64-x64.msi上传到云盘，以上文件截至2020年10月31日均为最新，需要的[自行下载](https://cloud.189.cn/t/f6nu6reeyUvi) 。

**使用**

首先克隆或者下载该库的 ZIP 文件，

#### 趣味实例-图像修复

---

###### **概述和环境准备**

图像修复使用的是 Github 上 shepnerd 的开源项目**[inpainting_gmcnn](https://github.com/shepnerd/inpainting_gmcnn)**，克隆或下载后通过以下链接下载预训练的模型（[paris_streetview](https://drive.google.com/file/d/1wgesxSUfKGyPwGQMw6IXZ9GLeZ7YNQxu/view?usp=sharing), [CelebA-HQ_256](https://drive.google.com/file/d/1zvMMzMCXNxzbYJ_6SEwt3hUShD3Xnz9W/view?usp=sharing), [CelebA-HQ_512](https://drive.google.com/file/d/1cp5e8XyXmHNZWj_piHH4eg4HFi3ICl0l/view?usp=sharing), [Places2](https://drive.google.com/file/d/1aakVS0CPML_Qg-PuXGE1Xaql96hNEKOU/view?usp=sharing)），然后解压缩并将其放入项目目录下tensorflow文件夹新建的checkpoints文件夹下，如果你已经安装了tensorflow并且版本为 2.X 则需要降低版本到1.4及以上版本（不包括 2.X ），并且需要确保已经安装了numpy,scipy,easydict

**打开GUI**

在tensorflow文件夹下打开cmd,输入 `python painter_gmcnn.py --load_model_dir ./checkpoints/places2_512x680_freeform --img_shapes 512,680`,其中，load_model_dir为模型文件，填写刚刚下载好并且解压保存在 checkpoints 文件夹下的模型文件夹，img_shapes表示上传的图片的长宽大小,输入后回车，显示如下GUI页面：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/fill_img_gui_1.png)

点击 load上传图片，图片的类型依据刚刚选择的模型的类型而定，否则结果将不准确，点击rectangle 或 stroke对左侧页面做出一些破坏，点击fill通过模型还原的结果会在右侧显示，如图：

上传的是这张图片：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/fill_img_gui_2.png)

破坏并修复后的效果：

![](https://github.com/OscarXsb/OpenCV-BasicFunc/blob/master/references/fill_img_gui_3.png)