import pandas as pd
import operator
import argparse
import numpy as np
from scipy.io import arff

args  =  None

# 初始化静态参数
def args_init_static():
    parser  =  argparse.ArgumentParser()
    parser.add_argument("--filename",          type = str,       default = "Dataset/weather.nominal.arff",     help = "")
    args  =  parser.parse_args()
    return args

class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data

    def __str__(self):
        print(self.col, self.value)
        print(self.results)
        print(self.summary)
        return ""


def calculateDiffCount(datas):
    # 将输入的数据汇总(input, dataSet)
    # return results Set{type1:type1Count, type2:type2Count .... typeN:typeNCount}
    """
    该函数是计算gini值的辅助函数，假设输入的dataSet为为['A', 'B', 'C', 'A', 'A', 'D']，
    则输出为['A':3,' B':1, 'C':1, 'D':1]，这样分类统计dataSet中每个类别的数量
    """
    results = {}
    for data in datas:
        # data[-1] means dataType
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1
    return results


# gini()
def gini(data):
    # 计算gini的值(Calculate GINI)

    length = len(data)
    results = calculateDiffCount(data)
    imp = 0.0
    for i in results:
        imp += results[i] / length * results[i] / length
    return 1 - imp

def splitDatas(data, value, column):
    # 根据条件分离数据集(splitDatas by value, column)
    # return 2 part（list1, list2）

    list1 = []
    list2 = []

    if isinstance(value, int) or isinstance(value, float):
        for row in data:
            if row[column] >= value:
                list1.append(row)
            else:
                list2.append(row)
    else:
        for row in data:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)
    return list1, list2

def buildDecisionTree(data):
    # 递归建立决策树， 当gain=0，时停止回归
    # build decision tree bu recursive function
    # stop recursive function when gain = 0
    # return tree
    currentGain = gini(data)
    column_lenght = len(data[0])
    rows_length = len(data)

    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for col in range(column_lenght - 1):
        col_value_set = set([x[col] for x in data])
        for value in col_value_set:
            list1, list2 = splitDatas(data, value, col)
            p = len(list1) / rows_length
            gain = currentGain - p * gini(list1) - (1 - p) * gini(list2)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)
    dcY = {'impurity': '%.3f' % currentGain, 'sample': '%d' % rows_length}
    #
    # stop or not stop

    if best_gain > 0:
        trueBranch = buildDecisionTree(best_set[0])
        falseBranch = buildDecisionTree(best_set[1])
        return Tree(col=best_value[0], value = best_value[1], trueBranch = trueBranch, falseBranch=falseBranch, summary=dcY)
    else:
        return Tree(results=calculateDiffCount(data), summary=dcY, data=data)


def prune(tree, miniGain):
    # 剪枝 when gain < mini Gain, 合并（merge the trueBranch and falseBranch）
    if tree.trueBranch.results == None:
        prune(tree.trueBranch, miniGain)
    if tree.falseBranch.results == None:
        prune(tree.falseBranch, miniGain)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.trueBranch.data)
        len2 = len(tree.falseBranch.data)
        len3 = len(tree.trueBranch.data + tree.falseBranch.data)

        p = float(len1) / (len1 + len2)

        gain = gini(tree.trueBranch.data + tree.falseBranch.data) - p * gini(tree.trueBranch.data) - (1 - p) * gini(tree.falseBranch.data)

        if gain < miniGain:
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculateDiffCount(tree.data)
            tree.trueBranch = None
            tree.falseBranch = None

def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


def load_data():
    df  =  arff.loadarff(args.filename)
    data  =  pd.DataFrame(df[0])
    data  =  np.array(data.values)
    return data

if __name__ == '__main__':
    args  =  args_init_static()
    dataSet = load_data()
    decisionTree = buildDecisionTree(dataSet)
    prune(decisionTree, 0.4)
    testDataSet = dataSet[:,0:-1]
    for testData in testDataSet:
        r = classify(testData, decisionTree)
        print(r)