import numpy as np
import random
import argparse
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

args = None

# 初始化静态参数
def args_init_static():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute_dim",     type=int,       default=8,       help="")
    parser.add_argument("--data_length",       type=int,       default=768,     help="")
    parser.add_argument("--epoch",             type=int,       default=300,     help="")
    parser.add_argument("--gamma",             type=int,       default=0.01,    help="")
    parser.add_argument("--filename",          type=str,       default="Dataset/diabetes.arff",     help="")
    args = parser.parse_args()
    return args

# sigmoid函数
def sigmoid(x):
    if x>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))
    # return 1.0/(1+np.exp(-x))

# 改进版随机梯度下降
def stocGradAscent(data, labels):
    m,n=np.shape(data)
    gamma = args.gamma
    weights=np.ones(args.attribute_dim+1)
    for j in range(args.epoch):
        for i in range(m):
            i = random.randint(0,args.data_length-1)
            h=sigmoid(weights.dot(data[i].T))
            error=labels[i]-h # 损失
            weights=weights+gamma*error*data[i] # 更新方程
    return weights

def test(weights, data, label):
    sum_num = 0
    all = 0
    for i in range(args.data_length):
        error = abs(sigmoid(sum(data[i]*weights)) - label[i])
        if error < 0.5:
            sum_num+=1
        all+=1
    rate = sum_num/all
    return sum_num, all, rate

def load_data():
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = np.array(data.values)
    return data

if __name__=='__main__':
    args = args_init_static()
    data = load_data()
    data = np.append(np.ones((args.data_length,1)),data,axis=1) # 属性前面加一个常数1，含义为偏置
    data2 = data[:,0:args.attribute_dim+1]
    labels = data[:,args.attribute_dim+1:]
    for i in range(args.data_length): # 处理标签，转化为int型
        c = labels[i][0]
        if labels[i][0]==b'tested_positive':
            labels[i] = 1
        else:
            labels[i] = 0
    weights=stocGradAscent(data2, labels)
    sum, all, rate = test(weights, data2, labels)
    print(' 成功:', sum, ' 总共:', all, ' 准确度为:%2f%%' % (rate*100))

    # draw(weights,data2,labels)