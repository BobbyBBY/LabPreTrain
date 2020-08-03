import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import argparse

args  =  None

# 初始化静态参数
def args_init_static():
    parser  =  argparse.ArgumentParser()
    parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
    args  =  parser.parse_args()
    return args


# 求协方差矩阵C的特征值和特征向量
def get_feature(x):
    ave = np.mean(x,axis=0)
    m, n = np.shape(x)
    ave = np.tile(ave, (m, 1))
    a = x-ave
    b = a.T
    # x_cov = np.cov((x-ave).T) # 求x的协方差矩阵
    x_cov = np.cov(b)
    featValue, featVec=  np.linalg.eig(x_cov)  # 求解协方差矩阵的特征值和特征向量
    m = featValue.shape[0]
    feat = np.hstack((featValue.reshape((m,1)), featVec))
    feat = feat[np.argsort(-featValue)] # 按照featValue进行从大到小排序
    return feat[:,0], feat[:,1:]
    
def paint_varience_(featValue):
    plt.figure()
    plt.plot(featValue, 'k')
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('explained_variance_', fontsize=16)
    plt.show()
            
def PCA(data):
    featValue, featVec  = get_feature(data)
    featValue_sum = sum(featValue)                         # 利用方差贡献度自动选择降维维度
    featValue_cons = featValue / featValue_sum
    varience_con = 0
    i = 0
    for i in range(featValue.shape[0]):
        varience_con += featValue_cons[i]       # 前i个方差贡献度之和
        if varience_con >= 0.99:
            break
        
    datai = featVec[0:i+1]                      # 取前i个特征向量
    output = np.dot(datai, np.transpose(data))     # 矩阵叉乘
    return np.transpose(output)

def draw(data):
    dim = data.shape[1]
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            plt.subplot(dim,dim,i*dim+j+1)
            # 挑选出前两个维度作为x轴和y轴，选择分类为色彩轴
            x_axis = data[:,i]
            y_axis = data[:,j]
            # c指定点的颜色，当c赋值为数值时，会根据值的不同自动着色
            plt.scatter(x_axis, y_axis)
    plt.show()

def load_data():
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = np.array(data.values)
    return data

if __name__ == '__main__':
    args = args_init_static()
    data = load_data()
    data = np.array(data[:,0:-1],dtype=np.float64)
    draw(data)
    output = PCA(data)
    draw(output)