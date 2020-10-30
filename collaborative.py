import heapq
import numpy as np
import pandas as pd
path = '/content/recolab-data/collaborative/'
M3=[]
def user_cf(A,B):
    C = np.dot(A,np.dot(B,np.dot(B.T,np.dot(A,B))))
    return C

def item_cf(A,B):
    C = np.dot(B,np.dot(A,np.dot(B.T,np.dot(B,A))))
    return C

with open(path+'items.txt') as fil:
    items = fil.read()
    items = items.split('\n')
items = items[:-1]
items = np.array(items)
items = np.reshape(items,(items.shape[0],1))

with open(path+'ratings.txt') as fil:
    rate = fil.read().split('\n')
    for row in rate[:-1]:
        row = row.rstrip().split(" ")
        row = [int(r) for r in row]
        
        M3.append(row)

rate = np.array(M3)
M1 = np.zeros((rate.shape[0],rate.shape[0]),dtype = 'int32')
M2 = np.zeros((rate.shape[1],rate.shape[1]),dtype = 'int32')
sum1 = np.reshape(np.sum(rate,axis = 1),(rate.shape[0],1))
sum2 = np.reshape(np.sum(rate,axis = 0),(rate.shape[1],1))


for i in range(rate.shape[0]):
    M1[i][i] = int(sum1[i][0])

for i in range(rate.shape[1]):
    M2[i][i] = int(sum2[i][0])


M1_inv = np.zeros((M1.shape[0],M1.shape[1]))
M2_inv = np.zeros((M2.shape[0],M2.shape[1]))
for i in range(M1.shape[0]):
    if M1[i,i] == 0:
        M1_inv[i,i] = 0
    else:
        M1_inv[i,i] = 1/M1[i,i]

for i in range(M2.shape[0]):
    if M2[i,i] == 0:
        M2_inv[i,i] = 0
    else:
        M2_inv[i,i] = 1/M2[i,i]

M1_inv = M1_inv ** (0.5)
M2_inv = M2_inv ** (0.5)

print('User recommendation')
user_r = user_cf(M1_inv,rate)

user_500 = user_r[499,:]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)
print('Top 100 recommendations')
for i in top_100:
    print(items[i])
print('Top 5 recommendations')
for i  in top_5:
    print(items[i])


with open(path+'orig.txt') as f:
    orig = f.read().split(' ')
    orig = [int(i) for i in orig]

c1 = 0
for i in top_100:
    if(orig[i] == 1):
        c1 += 1
print('Number of correct predictions in top 100')
print(c1)

c2 = 0
for i in top_5:
    if(orig[i] == 1):
        c2 += 1
print('Number of correct predictions in top 5')
print(c2)

print('Item recommendation')
item_r = item_cf(M2_inv,rate)


user_500 = item_r[499,:]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)
print('Top 100 recommendations')
for i in top_100:
    print(items[i])
print('Top 5 recommendations')
for i in top_5:
    print(items[i])


c1 = 0
for i in top_100:
    if(orig[i] == 1):
        c1 += 1

print('Number of correct predictions in top 100')
print(c1)

c2 = 0
for i in top_5:
    if(orig[i] == 1):
        c2 += 1
print('Number of correct predictions in top 5')
print(c2)
