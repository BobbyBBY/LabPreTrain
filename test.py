np = [1,2,3,4]
np2 = np[:]
dic = {1:1,2:2,3:4,'e':5}
dic[3]+=1
print(type(np))
print(type(np[0:1]))
print(np[0:0])
for i in range(4):
    print(i)