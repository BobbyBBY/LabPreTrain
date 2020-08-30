import numpy as np
import random
import tensorflow as tf
import sys

a = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
data2 = np.array(a)
data3=np.array([[1,2,3,4]])
data4=np.array([[1,2],[2,2],[1,3],[1,1]])
data5=np.array([[1,2,6,4]])
data6=np.array([[1,2]])
data7=np.array([[468.0,678.0,4.0,400.0]], dtype=np.float64)
print(data3)
# print(np.power((data3-data5), 2))
# print(np.sum(np.power((data3- data5), 2))/data3.shape[1])
# print(sum(data5))
# print(np.multiply(data3, data5))
# print(data6.T)
# data6 = data6.reshape(1,-1)
# print(data6)
# print(data6.T)


# print(np.random.rand(2,4))
# print(data3*np.sum(data4*data6,axis=1))
x = 1.0/(1+np.exp(-data7))
print(x)

# print(np.exp(-data7)/(1+np.exp(-data7)))
x = np.array([1]*100)
label = tf.convert_to_tensor(x)
print(label)
tensor = tf.range(10)
tf.print(tensor, output_stream=sys.stderr)