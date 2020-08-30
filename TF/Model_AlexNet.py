import tensorflow as tf
from tensorflow.keras import layers,Model

class Model_AlexNet(Model):
    def __init__(self, output_dim=10):
        super(Model_AlexNet, self).__init__(name='Model_AlexNet')
        self.output_dim = output_dim
        self.layer1 = layers.Conv2D(filters=96,kernel_size=11, strides=2, padding='same',activation='relu',input_shape=(28,28,1))
        # self.layer1 = layers.Conv2D(filters=96,kernel_size=11, strides=4, padding='same',activation='relu',input_shape=(227,227,3))
        self.layer2 = layers.MaxPool2D(pool_size=3,strides=2)
        self.layer3 = layers.Conv2D(filters=256,kernel_size=5, strides=1, padding='same',activation='relu')
        self.layer4 = layers.MaxPool2D(pool_size=3,strides=2)
        self.layer5 = layers.Conv2D(filters=384,kernel_size=3, strides=1, padding='same',activation='relu')
        self.layer6 = layers.Conv2D(filters=384,kernel_size=3, strides=1, padding='same',activation='relu')
        self.layer7 = layers.Conv2D(filters=256,kernel_size=3, strides=1, padding='same',activation='relu')
        self.layer8 = layers.MaxPool2D(pool_size=2,strides=2)
        # self.layer8 = layers.MaxPool2D(pool_size=3,strides=2)
        self.layer9 = layers.Flatten()
        self.layer10 = layers.Dense(2048, activation='relu')
        # self.layer10 = layers.Dense(4096, activation='relu')
        self.layer11 = layers.Dropout(0.5)
        self.layer12 = layers.Dense(2048, activation='relu')
        # self.layer12 = layers.Dense(4096, activation='relu')
        self.layer13 = layers.Dropout(0.5)
        self.layer14 = layers.Dense(self.output_dim, activation='softmax')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        out = self.layer14(x)
        return out