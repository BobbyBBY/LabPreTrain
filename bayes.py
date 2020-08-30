import pandas as pd 
import numpy as np 
from collections import defaultdict
import argparse
from scipy.io import arff
from sklearn.model_selection import train_test_split

args = None

# 初始化静态参数
def args_init_static():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
    parser.add_argument("--layers",            type=int,       default=5,                       help="")
    args = parser.parse_args()
    return args

def load_data():
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = np.array(data.values)
    return data[:,0:len(data[0])-1],np.reshape(data[:,len(data[0])-1:len(data[0])],len(data))

class NBClassifier(object):
    def __init__(self, layers = 5):
        self.y = [] # 标签集合
        self.x = [] # 每个属性的数值集合
        self.py = defaultdict(float) # 标签的概率分布
        self.pxy = defaultdict(dict) # 每个标签下的每个属性的概率分布
        self.n = layers # 分级的级数

    # 计算元素在列表中出现的频率
    def prob(self,element,arr):
        prob = 0.0
        for a in arr:
            if element == a:
                prob += 1/len(arr)
        if prob == 0.0:
            prob = 0.001
        return prob

    def get_set(self,x,y):
        self.y = list(set(y))
        for i in range(x.shape[1]):
            self.x.append(list(set(x[:,i])))# 记录下每一列的数值集

    def train(self,x,y):
        x = self.preprocess(x) 
        self.get_set(x,y)
        # 获取p(y)
        for yi in self.y:
            self.py[yi] = self.prob(yi,y)
        # 获取p(x|y)
        for yi in self.y:
            for i in range(x.shape[1]):
                sample = x[y==yi,i] # 标签yi下的样本
                # 获取该列的概率分布
                pxy = [self.prob(xi,sample) for xi in self.x[i]]
                self.pxy[yi][i] = pxy
        print("train score",self.score(x,y))

    # 预测单个样本
    def predict_one(self,x):
        max_prob = 0.0
        max_yi = self.y[0]
        for yi in self.y:
            prob_y = self.py[yi]
            for i in range(len(x)):
                if (x[i] in self.x[i]):
                    prob_x_y = self.pxy[yi][i][self.x[i].index(x[i])] # p(xi|y)
                else:
                    prob_x_y = 0
                prob_y *= prob_x_y # 计算p(x1|y)p(x2|y)...p(xn|y)p(y)
            if prob_y > max_prob:
                max_prob = prob_y
                max_yi = yi
        return max_yi

    # 预测函数
    def predict(self,samples):
        samples = self.preprocess(samples)
        y_list = []
        for m in range(samples.shape[0]):
            yi = self.predict_one(samples[m,:])
            y_list.append(yi)
        return np.array(y_list)

    # 因为不同特征的数值集大小相差巨大，造成部分概率矩阵变得稀疏，需要进行数据分割
    def preprocess(self,x):
        for i in range(x.shape[1]):
            x[:,i] = self.step(x[:,i],self.n)
        return x

    # 分为n阶
    def step(self,arr,n):
        ma = max(arr)
        mi = min(arr)
        for i in range(len(arr)):
            for j in range(n):
                a = mi + (ma-mi)*(j/n)
                b = mi + (ma-mi)*((j+1)/n)
                if arr[i] >= a and arr[i] <= b:
                    arr[i] = j+1
                    break
        return arr

    def score(self,x,y):
        y_test = self.predict(x)
        score = 0.0
        for i in range(len(y)):
            if y_test[i] == y[i]:
                score += 1/len(y)
        return score

if __name__ == "__main__":
    args = args_init_static()
    x,y = load_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
    clf = NBClassifier(args.layers)
    clf.train(x_train,y_train)
    score = clf.score(x_test,y_test)
    print('test score',score)
    print('test score',score)