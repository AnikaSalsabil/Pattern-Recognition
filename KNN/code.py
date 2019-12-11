"""
Anika Salsabil
AUST, CSE
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
 
# to read from a text file
#Train data
df=pd.read_csv('train.txt', sep=",", header= None, dtype='float')
df.columns=['Train-X', 'Train-Y', 'Class']
print('\n*****Train data*****')
print(df)
df_arr = df.values
trainlength= len(df_arr)

#Test data
dftest=pd.read_csv('test.txt', sep=",", header= None, dtype='float')
dftest.columns=['Test-X', 'Test-Y']
print('\n*****Test data*****')
print(dftest)
dftest_arr = dftest.values
testlength= len(dftest_arr)

#Task-1
X1, X2, Y1, Y2=[], [], [], []
for i in range(trainlength):
    if df_arr[i][2]==1:
        X1.append(df_arr[i][0])
        Y1.append(df_arr[i][1])
    else:
        X2.append(df_arr[i][0])
        Y2.append(df_arr[i][1])

plt.show()
plt.scatter(X1, Y1, color = 'red', marker = 'o')    
plt.scatter(X2, Y2, color = 'black', marker = 'o')  

#Task-2
dist1, dist2, x, y = [], [], [], []

#def KNN(train, test, k):
dis_list = []
class_list = []

for i in range(testlength):
	dw1 = []
	dw2 = []
	for j in range(trainlength):
		dis1 = (dftest_arr[i][0] - df_arr[j][0]) ** 2;
		dis2 = (dftest_arr[i][1] - df_arr[j][1]) ** 2;
		dw1.append(np.sqrt((dis1+dis2)));
		dw2.append(df_arr[j][2])

	dis_list.append(dw1)
	class_list.append(dw2)

#print("\n\nvalue= ", dis_list)
#print("\n\nvalue= ", class_list)          
        
k = int(input ("\nEnter the value of k :")) 
print("\nFor k =",k)


def bubbleSort(dis_list, class_list):
    for k in range(len(dis_list)-1,0,-1):
        for i in range(k):
            if dis_list[i] > dis_list[i+1]:
                temp = dis_list[i]
                dis_list[i] = dis_list[i+1]
                dis_list[i+1] = temp
                
                temp1 = class_list[i]
                class_list[i] = class_list[i+1]
                class_list[i+1] = temp1

disK=[]
clsK=[]
for i in range(len(dis_list)):
    bubbleSort(dis_list[i], class_list[i])
    for j in range(k):
        disK.append(dis_list[i][j])
        clsK.append(class_list[i][j])
        
disK_=np.resize(disK, (testlength, k))    
clsK_=np.resize(clsK, (testlength, k))   

"""    
print("\nDistance sorted=== ",dis_list)
print("\nClass sorted=== ",class_list)

print("\nDistance K=== ",disK_)
print("\nClass K=== ",clsK_)
"""

for i in range(testlength):
    for j in range(k):
        one=0
        two=0
        if clsK_[i][j]==1:
            one= one + 1
        else:
            two= two + 1
    
    if one > two:
        #print("Predicted Class: 1")
        plt.scatter(dftest_arr[i][0], dftest_arr[i][1], color = 'orange', marker = '*') 
        
    else:
        #print("Predicted Class: 2") 
        plt.scatter(dftest_arr[i][0], dftest_arr[i][1], color = 'blue', marker = '*') 
        
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Task-1 & 2')
colors = ['orange','blue']
lines = [Line2D([0], [0], color=c, linewidth=0, marker='*') for c in colors]
labels = ['Test C-1','Test C-2']
plt.legend(lines, labels) 
plt.show()

#Task-3
file = open("prediction.txt","w")     
for i in range(testlength):
     file.write("Test point: {}, {}\n" .format(dftest_arr[i][0], dftest_arr[i][1]))
     for j in range(k):
         one=0
         two=0
         file.write("Distance {}: {} \t\t\t\t Class: {}\n".format((j+1),disK_[i][j], clsK_[i][j]))
         
         if clsK_[i][j]==1:
            one= one + 1
         else:
            two= two + 1
     if one > two:
         file.write("Predicted Class: 1")
        
     else:
         file.write("Predicted Class: 2")  
         
     file.write("\n\n")
     
file.close()





