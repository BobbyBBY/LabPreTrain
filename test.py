import numpy as np
data = np.ones((4,1))
a = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
data2 = np.array(a)
data3=np.array([1,2,3,4])
data4=np.array([[1,2],[2,2],[3,3],[1,1]])
# data5 = data4.dot(data3)
data_1 = np.append(data,data2,axis=1)
data_2 = data_1[:,0:3]
print(data3)
print(data3.T)
# print(sum(data5))