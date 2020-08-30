import cv2
import tensorflow as tf
from tensorflow.keras import layers,Model
import numpy as np

class Model_CNN(Model):
    def __init__(self):
        super(Model_CNN, self).__init__(name='Model_CNN')
        self.layer1 = layers.Conv2D(filters=16,kernel_size=3, padding='SAME',activation='relu')
        self.layer2 = layers.Conv2D(filters=16,kernel_size=3, padding='SAME',activation='relu')
        self.layer3 = layers.MaxPool2D(pool_size=2,strides=2)
        self.layer4 = layers.Conv2D(filters=32,kernel_size=3,  padding='SAME',activation='relu')
        self.layer5 = layers.Conv2D(filters=32,kernel_size=3,  padding='SAME',activation='relu')
        self.layer6 = layers.MaxPool2D(pool_size=2,strides=2)
        self.layer7 = layers.Conv2D(filters=64,kernel_size=3, padding='SAME',activation='relu')
        self.layer8 = layers.Conv2D(filters=64,kernel_size=3,  padding='SAME',activation='relu')
        # self.layer9 = layers.MaxPool2D(pool_size=2,strides=2)
        # self.layer10 = layers.Conv2D(filters=128,kernel_size=3,  padding='SAME',activation='relu')
        # self.layer11 = layers.Conv2D(filters=128,kernel_size=3,  padding='SAME',activation='relu')
        self.layer12 = layers.Flatten()
        self.layer13 = layers.Dense(512, activation='relu')
        self.layer14 = layers.Dropout(0.2)
        self.layer15 = layers.Dense(256, activation='relu')
        self.layer16 = layers.Dropout(0.2)
        self.layer17 = layers.Dense(3, activation=None)
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # x = self.layer9(x)
        # x = self.layer10(x)
        # x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        out = self.layer17(x)
        return out

optimizer = tf.keras.optimizers.Adam(lr=0.0001)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model = Model_CNN()
model.compile(optimizer=optimizer, loss=loss_func, metrics=['categorical_accuracy'])#, metrics=['categorical_accuracy']
model.load_weights('model/CNN_model')


# vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\1598669367563.MP4")  # 读入视频文件
# vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\新视频.MP4")  # 读入视频文件
# vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\gitrevert.MP4")  # 读入视频文件
vc = cv2.VideoCapture(0)  # 打开摄像头
 
rval, firstFrame = vc.read()
firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_LINEAR)# interpolation：插值方法
gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)   # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)      #高斯模糊，用于去噪
prveFrame = firstFrame.copy()
 
#遍历视频的每一帧
while True:
    # ret表示是否正确读取到了帧
    # frame 每一帧，三维矩阵
    (ret, frame) = vc.read()
 
    # 如果没有获取到数据，则结束循环。如视频结束
    if not ret:
        break
 
    # 对获取到的数据进行预处理
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("oriegin_frame", frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # cv2.imshow("current_frame", gray_frame)
    # cv2.imshow("prveFrame", prveFrame)
 
    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(prveFrame, gray_frame)
    # cv2.imshow("frameDiff", frameDiff)
    prveFrame = gray_frame.copy()
 
    # 图像的二值化
    # 忽略较小的差别
    retVal, thresh = cv2.threshold(frameDiff, 20, 255, cv2.THRESH_BINARY)
 
 
    # 对阈值图像进行填充补洞
    thresh = cv2.dilate(thresh, None, iterations=20)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    text = "Unoccupied"
    # 遍历轮廓
    area = 0
    for contour in contours:
        # if contour is too small, just ignore it
        area_temp = cv2.contourArea(contour)
        if area_temp < area:
            continue
        if cv2.contourArea(contour) < 500:   #面积阈值
            continue
 
        # 计算最小外接矩形（非旋转）
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dst = frame[y: y + h, x: x + w]
        cv2.imshow('dst', dst)
        dst = tf.image.resize(dst,[100,100]) # 统一图片大小
        dst = tf.cast(dst,tf.float32) # 转换类型
        dst = dst/255 # 归一化
        dst = dst[tf.newaxis, ...]
        result_ori = model.predict(dst)
        result = np.argmax(result_ori)
        if result == 0:
            text = "dog"
        elif result == 1:
            text = "cat"
        else:
            text = "None"
        
 
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.putText(frame, "F{}".format(frameCount), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    cv2.imshow('frame_with_result', frame)
    cv2.imshow('thresh', thresh)
    # cv2.imshow('frameDiff', frameDiff)
 
    # 处理按键效果
    key = cv2.waitKey(60) & 0xff
    if key == 27:  # 按下ESC时，退出
        break
    elif key == ord(' '):  # 按下空格键时，暂停
        cv2.waitKey(0)
 
    cv2.waitKey(0)
 
vc.release()