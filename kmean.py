import copy
import matplotlib.pyplot as plt
import argparse
import math
from collections import defaultdict
import numpy as np
from scipy.io import arff
import pandas as pd
import random

args = None

def args_init_static():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_classes",         type=int,       default=3,    help="")
    parser.add_argument("--attribute_dim",     type=int,       default=4,     help="")
    parser.add_argument("--labels",            type=list,      default=["Iris-setosa","Iris-versicolor","Iris-virginica"],    help="")
    parser.add_argument("--data_length",       type=int,       default=150,     help="")
    parser.add_argument("--center_diff",       type=float,       default=0.001,     help="")
    parser.add_argument("--filename",          type=str,       default="Dataset/iris.arff",     help="")
    args = parser.parse_args()
    return args

def generateCenters(data):
    # 初始化聚类中心,人为选定
    centers = []
    centers.append(data[24][:4])
    centers.append(data[74][:4]) 
    centers.append(data[124][:4]) 
    return centers
 
def distance(a ,b):
    # 欧式距离
    sum = 0
    for i in range(args.attribute_dim):
        sq = (a[i]-b[i])*(a[i]-b[i])
        sum += sq
    return math.sqrt(sum)
 
def point_avg(points):
    # 对维度求平均值
    new_center = []
    for i in range(args.attribute_dim):
        sum = 0
        for p in points:
            sum += p[i]
        new_center.append(float("%.8f" % (sum/float(len(points)))))
    return new_center
 
def updataCenters(data, assigments):
    new_means = defaultdict(list)
    centers = []
    for assigment, point in zip(assigments, data):
        new_means[assigment].append(point)
        # 将同一类的数据进行整合
    for i in range(args.k_classes):
        points = new_means[i]
        centers.append(point_avg(points))
    return np.asarray(centers)
 
def assignment(data, centers):
    assignments = []
    for point in data:
        # 遍历所有数据
        shortest = float('inf')
        shortestindex = 0
        for i in range(args.k_classes):
            # 遍历三个中心向量，与哪个类中心欧氏距离最短就将其归为哪类
            value = distance(point, centers[i])
            if value < shortest:
                shortest = value
                shortestindex = i
        assignments.append(shortestindex)
    return assignments
 
def center_diff(a,b):
    if a is None or b is None :
        return True
    sum = 0
    for i in range(args.k_classes):
        sum+=abs(a[i]-b[i])
    sum = sum.sum()
    return sum > args.center_diff

def kmeans(data):
    new_centers = generateCenters(data)
    assigments = assignment(data, new_centers)
    centers = None
    # center_diff
    while center_diff(centers,new_centers):
        centers = new_centers
        new_centers = updataCenters(data, assigments)
        assigments = assignment(data, new_centers)
    result = copy.deepcopy(data)
    for i in range(args.data_length):
        result[i][4] = assigments[i]
    return result
 
def test(result):
    # 由于数据集根据花的种类排序，所以可以使用以下方式测试准确度
    sum = 0
    all = 0
    for j in range(args.k_classes):
        for i in range(50):
            if result[j*50+i][4] == j:
                sum += 1
            all += 1
    rate = sum/all
    return sum, all, rate

def draw(data):
    for i in range(args.attribute_dim):
        for j in range(args.attribute_dim):
            if i == j:
                continue
            plt.subplot(args.attribute_dim,args.attribute_dim,i*4+j+1)
            #挑选出前两个维度作为x轴和y轴，选择分类为色彩轴
            x_axis = data[:,i]
            y_axis = data[:,j]
            z_axis = data[:,4]
            #c指定点的颜色，当c赋值为数值时，会根据值的不同自动着色
            plt.scatter(x_axis, y_axis,c = z_axis)
    plt.show()

if __name__ == "__main__":
    args = args_init_static()
    df = arff.loadarff(args.filename)
    data = pd.DataFrame(df[0])
    data = data.values
    for i in range(3):
        for j in range(50):
            data[i*50+j][4]=i
    result = kmeans(data)
    for i in range(args.k_classes):
        tag = 0
        print('\n')
        print("%s类：" % (args.labels[i]))
        for tuple in range(len(result)):
            if(result[tuple][4] == i):
                print(tuple, end=' ')
                tag += 1
            if tag > 20 :
                print('\n')
                tag = 0
    print('\n')
    sum, all, rate = test(result)
    print('成功:', sum, '总共:', all)
    print('准确度为:%2f%%' % (rate*100))
    draw(data)
    draw(result)