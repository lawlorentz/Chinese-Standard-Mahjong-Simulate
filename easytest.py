# 对产生的147维feature正确性的检验
import numpy as np

testNO = 0
testRound = 0

def matrix(m):
    cards = []
    for i in range(4):
        for j in range(9):
            if m[i][j] == 1:
                if i == 0:
                    cards.append('W'+'%d' % (j+1))
                elif i == 1:
                    cards.append('T'+'%d' % (j+1))
                elif i == 2:
                    cards.append('B'+'%d' % (j+1))
                elif i == 3:
                    if j < 4:
                        cards.append("F%d" % (j+1))
                    else:
                        cards.append('J'+'%d' % (j-3))
    return cards

def matrix_chi(m):
    cards = []
    for i in range(4):
        for j in range(9):
            if m[i][j] == 1:
                if i == 0:
                    cards.append('W'+'%d' % (j))
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+2))
                elif i == 1:
                    cards.append('T'+'%d' % (j))
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+2))
                elif i == 2:
                    cards.append('B'+'%d' % (j))
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+2))
                elif i == 3: #no possible
                    if j < 4:
                        cards.append("F%d" % (j+1))
                    else:
                        cards.append('J'+'%d' % (j-3))
    return cards

def matrix_peng(m):
    cards = []
    for i in range(4):
        for j in range(9):
            if m[i][j] == 1:
                if i == 0:
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+1))
                elif i == 1:
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+1))
                elif i == 2:
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+1))
                elif i == 3:
                    if j < 4:
                        cards.append("F%d" % (j+1))
                        cards.append("F%d" % (j+1))
                        cards.append("F%d" % (j+1))
                    else:
                        cards.append('J'+'%d' % (j-3))
                        cards.append('J'+'%d' % (j-3))
                        cards.append('J'+'%d' % (j-3))
    return cards

def matrix_gang(m):
    cards = []
    for i in range(4):
        for j in range(9):
            if m[i][j] == 1:
                if i == 0:
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+1))
                    cards.append('W'+'%d' % (j+1))
                elif i == 1:
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+1))
                    cards.append('T'+'%d' % (j+1))
                elif i == 2:
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+1))
                    cards.append('B'+'%d' % (j+1))
                elif i == 3:
                    if j < 4:
                        cards.append("F%d" % (j+1))
                        cards.append("F%d" % (j+1))
                        cards.append("F%d" % (j+1))
                        cards.append("F%d" % (j+1))
                    else:
                        cards.append('J'+'%d' % (j-3))
                        cards.append('J'+'%d' % (j-3))
                        cards.append('J'+'%d' % (j-3))
                        cards.append('J'+'%d' % (j-3))
    return cards

a=np.load(r'国标麻将\data\%d.npz' % testNO)
b = a['obs']
cnum = len(b)
testRound = cnum -1

data2 = open(r'国标麻将\data\mt%d.txt' % testRound,'w+')
for i in range(147):
    print('\n%d\n' % i,file=data2)
    print(a['obs'][testRound][i],file=data2)
data2.close()

data=open(r'国标麻将\data\pp%d.txt' % testRound,'w+')

Feng_chan = matrix(a['obs'][testRound][0])
Feng_zi = matrix(a['obs'][testRound][1])

print('\nchangfen\n',file= data)
print(Feng_chan,file=data)
print('\nzifeng\n',file= data)
print(Feng_zi,file=data)

shoupai_1 = matrix(a['obs'][testRound][2])
shoupai_2 = matrix(a['obs'][testRound][3])
shoupai_3 = matrix(a['obs'][testRound][4])
shoupai_4 = matrix(a['obs'][testRound][5])
shoupai = shoupai_1+shoupai_2+shoupai_3+shoupai_4
shoupai.sort()

print('\nshoupai\n',file= data)
print(shoupai,file=data)

cardlist = []                  # 创建一个空列表
for i in range(30):      # 创建一个5行的列表（行）
    cardlist.append([]) 

p = 6
t = 4
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_chi(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia0 chi \n',file= data)
print(cardlist[0],file=data)
p = 10
t = 4
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_chi(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia1 chi \n',file= data)
print(cardlist[0],file=data)
p = 14
t = 4
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_chi(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia2 chi \n',file= data)
print(cardlist[0],file=data)
p = 18
t = 4
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_chi(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia3 chi \n',file= data)
print(cardlist[0],file=data)

p = 22
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_peng(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia0 peng \n',file= data)
print(cardlist[0],file=data)
p = 23
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_peng(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia1 peng \n',file= data)
print(cardlist[0],file=data)
p = 24
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_peng(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia2 peng \n',file= data)
print(cardlist[0],file=data)
p = 25
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_peng(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia3 peng \n',file= data)
print(cardlist[0],file=data)

p = 26
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_gang(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia0 gang \n',file= data)
print(cardlist[0],file=data)
p = 27
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_gang(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia1 gang \n',file= data)
print(cardlist[0],file=data)
p = 28
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_gang(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia2 gang \n',file= data)
print(cardlist[0],file=data)
p = 29
t = 1
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix_gang(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia3 gang \n',file= data)
print(cardlist[0],file=data)

p = 30
t = 28
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia0 qipai \n',file= data)
print(cardlist[0],file=data)
p = 58
t = 28
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia1 qipai \n',file= data)
print(cardlist[0],file=data)
p = 86
t = 28
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia2 qipai \n',file= data)
print(cardlist[0],file=data)
p = 114
t = 28
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nwanjia3 qipai \n',file= data)
print(cardlist[0],file=data)

p = 143
t = 4
cardlist[0] = []
for i in range(t):
    cardlist[i+1] = matrix(a['obs'][testRound][p+i])
    cardlist[0] += cardlist[i+1]
print('\nmeimodepai \n',file= data)
print(cardlist[0],file=data)

data.close()





