import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plot
from matplotlib import style

path = '/content/recolab-data/latent/train.tit'
train_data = []
test_data = []
with open(path) as f:
    data = f.read().split('\n')
    for line in data[:-1]:
        l = line.split('\t')

        train_data.append([int(i) for i in l])

train_data = np.array(train_data)

path = '/content/recolab-data/latent/test.tit'
with open(path) as f:
    data = f.read().split('\n')
    for line in data[:-1]:
        l = line.split('\t')

        test_data.append([int(i) for i in l])

test_data = np.array(test_data)

class Latent_Factor():
    def __init__(self,R,K,A,B,iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.A = A
        self.B = B
        self.iterations = iterations

    def train(self):
        self.P1 = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q1 = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        M = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            error = self.error()
            M.append((i, error))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, error))

        return M
    
    def error(self):
         xs, ys = self.R.nonzero()
         predicted = self.funct()
         error = 0
         for i, j in zip(xs, ys):
             error += pow(self.R[i, j] - predicted[i, j], 2)
         return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.find_prediction(i, j)
            err = 2*(r - prediction)
            
            
            self.P1[i, :] += self.A * (err * self.Q1[j, :] - 2*self.B * self.P1[i,:])
            self.Q1[j, :] += self.A * (err * self.P1[i, :] - 2*self.B * self.Q1[j,:])

    def find_prediction(self, i, j):
        prediction = self.P1[i, :].dot(self.Q1[j, :].T)
        return prediction

    def funct(self): 
        return self.P1.dot(self.Q1.T)


Z_matrix = np.max(train_data,axis = 0)
user = Z_matrix[1]
item = Z_matrix[0]
print(user,item)

R = np.zeros((user,item))

for i in train_data:
    R[i[1]-1,i[0]-1] = i[2]

 

print(R[:5]) 
fact = [10, 20, 40, 80, 100, 200]
for k in fact:
    A = 0.01
    B = 0.005
    lf = Latent_Factor(R, K=k, A=A, B=B, iterations=40)
    matrix = lf.train()
    error = 0.0
    predicted = lf.funct()
    for l in test_data:
        error += pow(l[2] - predicted[l[1]-1, l[0]-1], 2)
    print('Test Error : ',error)
    matrix = np.array(matrix)
    style.use('ggplot')
    plot.figure(figsize=(12,6))
    plot.xlabel("Iteration")
    plot.ylabel("Mean Square Error")
    s = "Latent Factor : "+str(k)+" lr = "+str(A) + ", lambda = " + str(B)
    plot.title(s)
    plot.plot(matrix[:,0],matrix[:,1])
    plot.show()
