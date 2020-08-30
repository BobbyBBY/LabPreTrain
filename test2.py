import  tensorflow as tf

import glob
import matplotlib.pyplot as plt
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_path = glob.glob('C:\\DesktopOthers\\Github\\LabPreTrain\\Dataset\\training_set\\training_set\\*.jpg') # 图片的路径

labels = []
for i in range(len(input_path)):
    if "dog" in input_path[i]:
        labels.append([1,0,0])
    else:
        labels.append([0,1,0])

def load_preprosess_image(input_path,label):
    
    image = tf.io.read_file(input_path) # 读取的是二进制格式 需要进行解码
    image = tf.image.decode_jpeg(image,channels=3)  # 解码 是通道数为3
    image = tf.image.resize(image,[256,256]) # 统一图片大小
    image = tf.cast(image,tf.float32) # 转换类型
    image = image/255 # 归一化
    
    return image, label  # return回的都是一个batch一个batch的 ， 一个批次很多张

train_dataset = tf.data.Dataset.from_tensor_slices((input_path,labels))
train_dataset = train_dataset.map(load_preprosess_image)  #.map是使函数应用在load_preprosess_image中所有的图像上 
# train_dataset = train_dataset.batch(32)

# train_dataset.unbatch()
imgs,las = next(iter(train_dataset)) #取出的是一个batch个数的图片 shape = (batch_size,256,256,3)

print(train_dataset[0])

plt.imshow(imgs[1]) # 展示我们读到的图像 
plt.show() 