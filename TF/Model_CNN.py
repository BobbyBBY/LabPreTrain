import tensorflow as tf
from tensorflow.keras import layers,Model

class Model_CNN(Model):
    def __init__(self):
        super(Model_CNN, self).__init__(name='Model_CNN')
        self.layer1 = layers.Conv2D(filters=6,kernel_size=5, strides=1, padding='SAME',activation='sigmoid',input_shape=(28,28,1))
        self.layer2 = layers.MaxPool2D(pool_size=2,strides=2)
        self.layer3 = layers.Conv2D(filters=16,kernel_size=5, strides=1, padding='SAME',activation='sigmoid')
        self.layer4 = layers.MaxPool2D(pool_size=2,strides=2)
        self.layer5 = layers.Flatten()
        self.layer6 = layers.Dense(120, activation='sigmoid')
        self.layer7 = layers.Dense(10, activation=None)
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        out = self.layer7(x)
        return out