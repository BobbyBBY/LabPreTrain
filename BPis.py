import numpy as np
import argparse
from scipy.io import arff
import pandas as pd

args = None

# 初始化静态参数
def args_init_static():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute_dim",     type=int,       default=4,       help="")
    parser.add_argument("--class_dim",         type=int,       default=3,       help="")
    parser.add_argument("--layers",            type=list,      default=[4,4,3],       help="")
    parser.add_argument("--data_length",       type=int,       default=150,     help="")
    parser.add_argument("--epoch",             type=int,       default=800,     help="")
    parser.add_argument("--lr",                type=int,       default=0.001,    help="")
    parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
    args = parser.parse_args()
    return args

# layers表示从输入层到输出层，每层神经元数量，这里规定为三层，目前无法实现更多层
def parameter_initialization(layers):
    length = len(layers)
    values = [] # 阈值
    weights = [] # 权重
    for i in range(length-1):
        values.append(np.random.rand(1,layers[i+1]))
        weights.append(np.random.rand(layers[i],layers[i+1]))
    return [weights,values]

# sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# 损失函数，平方损失函数
def square_loss( x, y, l):
    return np.sqrt(np.sum(np.power((x - y), 2))/l)

def stoc_grad_ascent(data, labels, net):
    lr = args.lr
    weights = net[0]
    values = net[1]
    net_length = len(weights)
    loss_all = 0
    for i in range(len(data)):
        input = []
        output = []
        # 输入层
        input.append(data[i:i+1])
        output.append(data[i:i+1])
        # 隐藏层、输出层
        for j in range(net_length):
            input.append(np.dot(output[j], weights[j]))
            output.append(sigmoid(input[j+1] - values[j]))
        loss = square_loss(output[-1], labels[i:i+1], output[-1].shape[1])
        # loss_all+=loss

        # 更新公式由矩阵运算表示
        l = output[-1].shape[1]
        temp1 = np.sqrt(np.power(loss, 2) * l * l)
        temp2 = output[-1] -labels[i]
        temp3 = output[-1] * (1 - output[-1]) * (temp2)
        dv2 = (lr / temp1) * temp3
        dw2 = -1 * np.dot(output[-2].T, dv2)
        dv1 = (lr / temp1) * output[-2] * (1 - output[-2]) * np.sum((weights[-1] * temp2), axis = 1)
        dw1 = -1 * np.dot(output[-3].T, dv1)
        #更新参数
        values[0] += dv1
        values[1] += dv2
        weights[0] += dw1
        weights[1] += dw2
    # print("loss_all: ", loss_all)
    return [weights,values]

def testing(data, labels, net):
    # 记录预测正确的个数
    sum_num = 0
    all = 0
    weights = net[0]
    values = net[1]
    net_length = len(weights)
    for i in range(len(data)):
        input = []
        output = []
        all += 1
        # 输入层
        input.append(data[i:i+1])
        output.append(data[i:i+1])
        # 隐藏层、输出层
        for j in range(net_length):
            input.append(np.dot(output[j], weights[j]))
            output.append(sigmoid(input[j+1] - values[j]))

        if np.argmax(output[-1]) == np.argmax(labels[i]):
            sum_num += 1
    rate = sum_num/all
    return sum_num, all, rate

def load_data():
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = np.array(data.values)
    return data

if __name__ == '__main__':
    args = args_init_static()
    data = load_data()
    data2 = np.array(data[:,0:args.attribute_dim], dtype=np.float64)
    labels = data[:,args.attribute_dim:]
    labels2 = np.zeros((args.data_length,args.class_dim), dtype=np.float64)
    for i in range(args.data_length): # 处理标签，转化为int型
        if labels[i][0]==b'Iris-setosa':
            labels2[i][0] = 1
        elif labels[i][0]==b'Iris-versicolor':
            labels2[i][1] = 1
        else:
            labels2[i][2] = 1
    for j in range(5):
        net = parameter_initialization(args.layers)
        for i in range(args.epoch):
            net = stoc_grad_ascent(data2, labels2, net)
        sum_num, all, rate = testing(data2, labels2, net)
        print(' 成功:', sum_num, ' 总共:', all, ' 准确度为:%2f%%' % (rate*100))