import numpy as np
import argparse
import random
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

args = None

# 初始化静态参数
def args_init_static():
     parser = argparse.ArgumentParser()
     parser.add_argument("--attribute_dim",     type=int,       default=4,       help="")
     parser.add_argument("--k",                 type=int,       default=3,       help="")
     parser.add_argument("--data_length",       type=int,       default=150,     help="")
     parser.add_argument("--epoch",             type=int,       default=300,     help="")
     parser.add_argument("--gamma",             type=int,       default=0.01,    help="")
     parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
     args = parser.parse_args()
     return args

# softmax函数，将线性回归值转化为概率的激活函数
def softmax(x):
     return np.exp(x) / np.sum(np.exp(x))

# 随机梯度下降
def stocGradAscent(data, labels):
     m,n=np.shape(data)
     gamma = args.gamma
     weights=np.ones((args.k,args.attribute_dim+1))
     for j in range(args.epoch):
          for i in range(m):
               i = random.randint(0,args.data_length-1)
               h=softmax(weights.dot(data[i]))
               error = labels[i]-h # 损失
               error = error.reshape(error.shape[0],1)
               weights=weights+gamma*error.dot(data[i:i+1]) # 更新方程
     return weights
     

def test(weights, data, labels):
     sum_num = 0
     all = 0
     for i in range(args.data_length):
          h=softmax(weights.dot(data[i]))
          predict = np.argmax(h)
          if 1 == labels[i][predict]:
               sum_num+=1
          all+=1
     rate = sum_num/all
     return sum_num, all, rate

def load_data():
     df = arff.loadarff(args.filename)
     data = pd.DataFrame(df[0])
     data = np.array(data.values)
     return data

if __name__ == "__main__":
     args = args_init_static()
     data = load_data()
     data = np.append(np.ones((args.data_length,1)),data,axis=1) # 属性前面加一个常数1，含义为偏置
     data2 = np.array(data[:,0:args.attribute_dim+1],dtype=np.float64)
     labels_str = data[:,args.attribute_dim+1:]
     labels = np.zeros((args.data_length,args.k),dtype=np.float64)
     for i in range(args.data_length): # 处理标签，转化为int型
          if labels_str[i][0]==b'Iris-setosa':
               labels[i][0] = 1
          elif labels_str[i][0]==b'Iris-versicolor':
               labels[i][1] = 1
          else :
               labels[i][2] = 1
     for i in range(15):
          weights=stocGradAscent(data2, labels)
          sum_num, all, rate = test(weights, data2, labels)
          print(' 成功:', sum_num, ' 总共:', all, ' 准确度为:%2f%%' % (rate*100))