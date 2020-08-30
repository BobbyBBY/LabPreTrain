import tensorflow as tf
from tensorflow.keras import layers,Model

class InceptionBlk(Model):
 
    def __init__(self, filters):
        super(InceptionBlk, self).__init__(name='InceptionBlk')
 
        self.conv1 = layers.Conv2D(filters=filters[0],kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_1 = layers.Conv2D(filters=filters[1],kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(filters=filters[2],kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3_1 = layers.Conv2D(filters=filters[3],kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(filters=filters[4],kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool4_1 = layers.MaxPooling2D(pool_size=3,strides=1, padding='same')
        self.conv4_2 = layers.Conv2D(filters=filters[5],kernel_size=1, strides=1, padding='same', activation='relu')
 
    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x4 = self.pool4_1(x)
        x4 = self.conv4_2(x4)
        out = tf.concat([x1, x2, x3, x4], axis=3)
        return out

class Support(Model):

    def __init__(self, output_dim=10):
        super(Support, self).__init__(name='Support')
        self.output_dim = output_dim

        self.pool1 = layers.AveragePooling2D(pool_size=5,strides=3, padding='same')
        self.conv2 = layers.Conv2D(filters=128,kernel_size=1, strides=1, padding='same', activation='relu')
        self.linear3 = layers.Dense(1024, activation='relu')
        self.drop4 = layers.Dropout(0.7)
        self.linear5 = layers.Dense(self.output_dim, activation='softmax')

    def call(self, x):
        x1 = self.pool1(x)
        x2 = self.conv2(x1)
        x3 = self.linear3(x2)
        x4 = self.drop4(x3)
        out = self.linear5(x4)
        return out

class Model_GoogLeNet(Model):
 
    def __init__(self, output_dim=10):
        super(Model_GoogLeNet, self).__init__(name='Model_GoogLeNet')
        self.output_dim = output_dim

        self.conv1 = layers.Conv2D(64,kernel_size=7, strides=2, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(pool_size=3,strides=2, padding='same')
        self.norm3 = layers.BatchNormalization()
        self.conv4_1 = layers.Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv4_2 = layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu')
        
        self.norm5 = layers.BatchNormalization()
        self.pool6 = layers.MaxPooling2D(pool_size=3,strides=2, padding='same')
        self.block7 = InceptionBlk([64,96,128,16,32,32])
        self.block8 = InceptionBlk([128,128,192,32,96,64])
        self.pool9 = layers.MaxPooling2D(pool_size=3,strides=2, padding='same')
        
        self.block10 = InceptionBlk([192,96,208,16,48,64])
        self.block11 = InceptionBlk([160,112,224,24,64,64])
        self.block12 = InceptionBlk([128,128,256,24,64,64])
        self.block13 = InceptionBlk([112,144,288,32,64,64])
        self.block14 = InceptionBlk([256,160,320,32,128,128])
        
        self.pool15 = layers.MaxPooling2D(pool_size=3,strides=2, padding='same')
        self.block16 = InceptionBlk([256,160,320,32,128,128])
        self.block17 = InceptionBlk([384,192,384,48,128,128])
        self.pool18 = layers.AveragePooling2D(pool_size=7,strides=1, padding='same')
        self.drop19 = layers.Dropout(0.4)

        self.linear20 = layers.Dense(1000, activation='relu')
        self.linear21 = layers.Dense(self.output_dim, activation='softmax')

        self.flat22 = layers.Flatten()

        self.support1 = Support(self.output_dim)
        self.support2 = Support(self.output_dim)

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.pool2(x1)
        x3 = self.norm3(x2)
        x4 = self.conv4_1(x3)
        x5 = self.conv4_2(x4)
        
        x6 = self.norm5(x5)
        x7 = self.pool6(x6)
        x8 = self.block7(x7)
        x9 = self.block8(x8)
        x10 = self.pool9(x9)

        x11 = self.block10(x10)
        x12 = self.block11(x11)
        x13 = self.block12(x12)
        x14 = self.block13(x13)
        x15 = self.block14(x14)

        x16 = self.pool15(x15)
        x17 = self.block16(x16)
        x18 = self.block17(x17)
        x19 = self.pool18(x18)
        x20 = self.drop19(x19)

        x21 = self.linear20(x20)
        out1 = self.linear21(x21)

        out2 = self.support1(x10)
        out3 = self.support2(x14)

        out = 0.4 * out1 + 0.3 * out2 + 0.3 * out3

        out = self.flat22(out)

        return out
