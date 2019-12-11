"""Anika Salsabil
AUST
CSE"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.lines import Line2D

df=pd.read_csv('data.txt', sep=" ", header= None, dtype='float')
df.columns=['Train-X', 'Train-Y']
print('\n*****Train data*****')
df_arr = df.values
trainlength= len(df_arr)

#Task-1
X, Y= [], []
for i in range(trainlength):
    X.append(df_arr[i][0])
    Y.append(df_arr[i][1])
plt.show()
plt.title('Datapoints plotted Graph')
plt.scatter(X, Y, color = 'black', marker = '.')  

#functions
def euclidean(x, y):
    distance=[]
    for i in range(trainlength):
        p = (df_arr[i][0] - x) ** 2
        q = (df_arr[i][1] - y) ** 2
        dis=np.sqrt((p+q))
        distance.append(dis)      
    return distance 

def kmeans_compare(dis1, dis2):
    X1, X2, Y1, Y2= [], [], [], []
    for i in range(trainlength):
        if dis1[i] < dis2[i]:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])
    return X1, X2, Y1, Y2

def kmeans(X1, X2, Y1, Y2):
    count=0
    mean_x1= np.mean(X1)
    mean_y1= np.mean(Y1)
    mean_x2= np.mean(X2)
    mean_y2= np.mean(Y2)
    
    up_dis1=euclidean(mean_x1, mean_y1)
    up_dis2=euclidean(mean_x2, mean_y2)
    
    upX1, upX2, upY1, upY2= kmeans_compare(up_dis1,up_dis2) 
    
    if X1==upX1 and X2==upX2 and Y1==upY1 and Y2==upY2:
        return upX1, upX2, upY1, upY2, mean_x1, mean_x2, mean_y1, mean_y2, count
    
    else:
        kmeans(upX1, upX2, upY1, upY2)
        count=count+1
    return upX1, upX2, upY1, upY2, mean_x1, mean_x2, mean_y1, mean_y2, count

#initial stage
cx,cy=[],[]
k=2

np.random.seed(123)
a=random.sample(range(0, 2999), k)
for i in range(len(a)):
    cx.append(X[a[i]])
    cy.append(Y[a[i]])
   
print("\nRandomly chosen:")
print("\nInitial centroid: x=",cx)   
print("Initial centroid: y=",cy) 
plt.scatter(cx, cy, color = 'red', marker = '*') 

ini_dis1, ini_dis2 = [], []
ini_dis1=euclidean(cx[0], cy[0])
ini_dis2=euclidean(cx[1], cy[1])

X1, X2, Y1, Y2= [], [], [], []
X1, X2, Y1, Y2= kmeans_compare(ini_dis1,ini_dis2)   

plt.show()
plt.title('Initial Graph')
plt.scatter(X1, Y1, color = 'red', marker = '.')  
plt.scatter(X2, Y2, color = 'green', marker = '.')  
plt.scatter(cx, cy, color = 'black', marker = 'o') 

#update stage
upX1, upX2, upY1, upY2, mean_x1, mean_x2, mean_y1, mean_y2, count= kmeans(X1, X2, Y1, Y2)  

plt.show()
plt.title('Updated Graph')
plt.scatter(upX1, upY1, color = 'red', marker = '.')  
plt.scatter(upX2, upY2, color = 'green', marker = '.')  
#plt.scatter(mean_x1, mean_y1, color = 'black', marker = 'o') 
#plt.scatter(mean_x2, mean_y2, color = 'black', marker = 'o') 

plt.xlabel('X')
plt.ylabel('Y')
colors = ['red','green']
lines = [Line2D([0], [0], color=c, linewidth=0, marker='o') for c in colors]
labels = ['Class C-1','Class C-2']
plt.legend(lines, labels) 
#print("The no of iterations: ", count)

    
