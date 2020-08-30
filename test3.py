import tensorflow as tf
from tensorflow.keras import layers,Model
import numpy as np
import os
import glob

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

def get_files(input_path):
    input_path = glob.glob(input_path) # 图片的路径
    labels = []
    for i in range(len(input_path)):
        if "dog" in input_path[i]:
            labels.append([1,0,0])
        elif "dog" in input_path[i]:
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
    return input_path, labels

def get_batch(input_path, labels):
    def load_preprosess_image(input_path,label):
        image = tf.io.read_file(input_path) # 读取的是二进制格式 需要进行解码
        image = tf.image.decode_jpeg(image,channels=3)  # 解码 是通道数为3
        image = tf.image.resize(image,[100,100]) # 统一图片大小
        # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
        # image = tf.image.resize_image_with_crop_or_pad(image, 150, 150)
        image = tf.cast(image,tf.float32) # 转换类型
        image = image/255 # 归一化
        return image, label  # return回的都是一个batch一个batch的 ， 一个批次很多张

    train_dataset = tf.data.Dataset.from_tensor_slices((input_path,labels))
    train_dataset = train_dataset.map(load_preprosess_image)  #.map是使函数应用在load_preprosess_image中所有的图像上 
    train_dataset = train_dataset.shuffle(len(input_path)).batch(32)
    return train_dataset

train_dir = "C:\\DesktopOthers\\Github\\LabPreTrain\\Dataset\\training_set\\training_set\\*.jpg"
test_dir_dogs = "C:\\DesktopOthers\\Github\\LabPreTrain\\Dataset\\test_set\\test_set\\dogs\\*.jpg"
test_dir_cats = "C:\\DesktopOthers\\Github\\LabPreTrain\\Dataset\\test_set\\test_set\\cats\\*.jpg"
test_dir_other = "C:\\DesktopOthers\\青海照片\\喻子钊\\*.jpg"

input_path, labels = get_files(test_dir_other)
# train_ds = get_batch(input_path, labels)
# test_ds_dogs = get_files(test_dir_dogs)
# test_ds_cats = get_files(test_dir_cats)

for i in range(len(input_path)):
    input_path[i] = tf.io.read_file(input_path[i]) # 读取的是二进制格式 需要进行解码
    input_path[i] = tf.image.decode_jpeg(input_path[i],channels=3)  # 解码 是通道数为3
    input_path[i] = tf.image.resize(input_path[i],[100,100]) # 统一图片大小
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    # image = tf.image.resize_image_with_crop_or_pad(image, 150, 150)
    input_path[i] = tf.cast(input_path[i],tf.float32) # 转换类型
    input_path[i] = input_path[i]/255 # 归一化
x = tf.convert_to_tensor(input_path)
y = tf.convert_to_tensor(labels)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0,nesterov=False)
# 稳定、收敛快且精度高于SGD
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 设置回调功能
filepath = 'model/CNN_model' # 保存模型地址
saved_model = tf.keras.callbacks.ModelCheckpoint(filepath, verbose = 1) # 回调保存模型功能
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'model/log') # 回调可视化数据功能

model = Model_CNN()
model.compile(optimizer=optimizer, loss=loss_func, metrics=['categorical_accuracy'])#, metrics=['categorical_accuracy']
model.load_weights('model/CNN_model')
# model.fit(train_ds, batch_size=32, epochs=5,callbacks = [saved_model, tensorboard])
# model.fit(train_ds, batch_size=32, epochs=5)
temp = model.evaluate(x,y,batch_size=32)

print(temp)

