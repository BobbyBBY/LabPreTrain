import argparse
import math
import numpy as np
from collections import defaultdict
from scipy.io import arff
import pandas as pd

args = None

# 初始化静态参数
def args_init_static():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute_dim",     type=int,       default=4,       help="")
    parser.add_argument("--data_length",       type=int,       default=150,     help="")
    parser.add_argument("--k_max",             type=int,       default=25,      help="")
    parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
    args = parser.parse_args()
    return args

# 计算欧式距离
def distance(a ,b):
    sum = 0
    for i in range(args.attribute_dim):
        sq = (a[i]-b[i])*(a[i]-b[i])
        sum += sq
    return math.sqrt(sum)

# 冒泡排序
def bubble(data):
    for i in range(len(data) - 1):  
        flag = True  # 交换标志位
        for j in range(len(data) - i - 1):  
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
                flag = False
        if flag:
            # 已经全部有序
            return data
    return data

# 计算所有点到该点的距离
def assignment(data, target):
    distances = []
    length = len(data)
    for i in range(length):
        value = distance(data[i], target)
        distance_list = [value,i]
        distances.append(distance_list)
    # 排序
    bubble(distances)
    return distances

# knn归类
def knn(data, target, k):
    dict = defaultdict(int)
    # 取前k个点
    distances = assignment(data, target)[0:k]
    label = None
    max = 0
    temp = 0
    # 计算这些点中哪个类别的最多
    for distance in distances:
        label_temp = data[distance[1]][4]
        temp = dict[label_temp]+1
        # 这里不能有等号，若两类别数量相等，则关注谁更解决该点，由于distances有序，则谁先抵达max谁更近
        if temp > max:
            label = label_temp
            max = temp
    return label
 
def test(data, result):
    sum = 0
    all = 0
    for i in range(args.data_length):
        if data[i][4] == result[i]:
            sum+=1
        all+=1
    rate = sum/all
    return sum, all, rate

def load_data():
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = np.array(data.values)
    return data

if __name__ == "__main__":
    args = args_init_static()
    data = load_data()
    for k in range(args.k_max):
        result = []
        for i in range(args.data_length):
            a = data[0:i]
            b = data[i+1:args.data_length]
            data_1 = np.append(a,b,axis=0)
            result.append(knn(data_1, data[i], k))
        sum, all, rate = test(data, result)
        print('k:',k,' 成功:', sum, ' 总共:', all, ' 准确度为:%2f%%' % (rate*100))