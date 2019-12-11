""" Name: Anika Salsabil
    AUST, CSE
"""
"""To run this code, at first do the following steps please: 
--Search "Anaconda Prompt" from the my start button
--Type the below command and proceed by giving "y"
----------------------------------------
conda install -c conda-forge prettytable
----------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as st
from prettytable import PrettyTable

# to read from a text file
#Train data
df=pd.read_csv('train.txt', sep=" ", header= None, dtype='float')
df.columns=['X', 'Y', 'Class']
print('\nTrain data')
print(df)
df_arr = df.values
length= len(df_arr)

#Task-1

X1, X2, Y1, Y2=[], [], [], []

for i in range(length):
    if df_arr[i][2]==1:
        X1.append(df_arr[i][0])
        Y1.append(df_arr[i][1])
    else:
        X2.append(df_arr[i][0])
        Y2.append(df_arr[i][1])

plt.show()
plt.scatter(X1, Y1, color = 'red', marker = '*')    #class1 Red marker '*'
plt.scatter(X2, Y2, color = 'black', marker = '^')  #class2 Black marker '^'

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Task-1')
plt.gca().legend(('Class-1','Class-2'))
plt.savefig('Task_1.png')

#Task-2

Y=[]
for i in range(length):
    if df_arr[i][2]==1:
        Y.append( np.dot(df_arr[i][0], df_arr[i][0]) )
        Y.append( np.dot(df_arr[i][1], df_arr[i][1]) )
        Y.append( np.dot(df_arr[i][0], df_arr[i][1]) )
        Y.append(df_arr[i][0])
        Y.append(df_arr[i][1])
        Y.append(1)
        
#negating Class-2-----Normalization
    else:
        Y.append( np.negative(np.dot(df_arr[i][0], df_arr[i][0])) )
        Y.append( np.negative(np.dot(df_arr[i][1], df_arr[i][1])) )
        Y.append( np.negative(np.dot(df_arr[i][0], df_arr[i][1])) )
        Y.append( np.negative(df_arr[i][0]) )
        Y.append( np.negative(df_arr[i][1]) )
        Y.append(-1)

Y=np.resize(Y, (length, 6)) 
print("\n\t\tTask-2")   
print("\nHigher Dimension, Y= \n",Y)


#Task-3
#Single processing(One at a Time)
def single(w, alpha):
    flag, count, limit = 0, 0, 200

    for count in range(limit):
        flag=0
        for row in Y:
            y=np.array(row)
            
            wT= w.transpose()
            g= np.dot(y, wT)

            if g <= 0:
                w = w + alpha * y         
            else:
                flag= flag+1
                
        count= count+1

        if flag==6:
            #print(w)
            return count
            break


#Batch Processing(Many At a time)
def batch(w, alpha):
    flag, count, limit = 0, 0, 200
    temp=np.zeros(6)
    
    for count in range(limit):
        flag=0
        
        for row in Y:
            y=np.array(row)
            
            wT= w.transpose()
            g= np.dot(y, wT)
            
            if g <= 0:
                temp= temp+ y
                        
            else:
                flag= flag+1
                
        count= count+1
        w = w + alpha * temp
        
        if flag==6:
            return count
            break
     
#Task-4
            
alphalist= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#All ones
singlelist, batchlist,singlelist_zero,batchlist_zero, singlelist_random, batchlist_random=[],[],[],[],[],[]
w=np.ones(6)
table= PrettyTable(['Value of Alpha ( Learning Rate )', 'One at a Time','Many at a Time'])

for i in range(len(alphalist)):
    singlelist.append(single(w, alphalist[i]))
    batchlist.append(batch(w, alphalist[i]))
    table.add_row([ alphalist[i], singlelist[i], batchlist[i] ])
    
print("\n\t\tCase 1: Initial Weight Vector All One ")
print(table)


#All zeros
w_zeros=np.zeros(6)
table_zero= PrettyTable(['Value of Alpha ( Learning Rate )', 'One at a Time','Many at a Time'])
for i in range(len(alphalist)):
    singlelist_zero.append(single(w_zeros, alphalist[i]))
    batchlist_zero.append(batch(w_zeros, alphalist[i]))
    table_zero.add_row([ alphalist[i], singlelist_zero[i], batchlist_zero[i] ])

print("\n\t\tCase 2: Initial Weight Vector All Zeros ")
print(table_zero)


#All randoms
np.random.seed(123)
w_randoms= np.random.random(np.array(6))
table_random= PrettyTable(['Value of Alpha ( Learning Rate )', 'One at a Time','Many at a Time'])
for i in range(len(alphalist)):
    singlelist_random.append(single(w_randoms, alphalist[i]))
    batchlist_random.append(batch(w_randoms, alphalist[i]))
    table_random.add_row([ alphalist[i], singlelist_random[i], batchlist_random[i] ])

print("\n\t\tCase 3: Initial Weight Vector All Randoms ")
print(table_random)

######

#Graph Plotting
#All Ones
plt.show()
plt.figure(figsize=(9,5))

xpos=np.arange(len(alphalist))
plt.xticks(xpos, alphalist)

plt.bar(xpos - 0.07, singlelist, width=0.2, label="One At a Time", color = 'blue')
plt.bar(xpos + 0.07, batchlist, width=0.2, label="Many At a Time", color = 'orange')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Task-4, Perceptron algorithm (Initial Weight: All Ones)')
plt.gca().legend(('One At a Time', 'Many At a Time'))
plt.savefig('Allones_graph.png')



#All zeros
plt.show()
plt.figure(figsize=(9,5))

xpos=np.arange(len(alphalist))
plt.xticks(xpos, alphalist)

plt.bar(xpos - 0.07, singlelist_zero, width=0.2, label="One At a Time", color = 'green')
plt.bar(xpos + 0.07, batchlist_zero, width=0.2, label="Many At a Time", color = 'red')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Task-4, Perceptron algorithm (Initial Weight: All Zeros)')
plt.gca().legend(('One At a Time', 'Many At a Time'))
plt.savefig('Allzeros_graph.png')

#All randoms
plt.show()
plt.figure(figsize=(9,5))

xpos=np.arange(len(alphalist))
plt.xticks(xpos, alphalist)

plt.bar(xpos - 0.07, singlelist_random, width=0.2, label="One At a Time", color = 'grey')
plt.bar(xpos + 0.07, batchlist_random, width=0.2, label="Many At a Time", color = 'yellow')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Task-4, Perceptron algorithm (Initial Weight: All Randoms)')
plt.gca().legend(('One At a Time', 'Many At a Time'))
plt.savefig('Allrandoms_graph.png')


