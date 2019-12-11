"""
Name: Anika Salsabil
AUST, CSE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as st

# to read from a text file
#Train data
df=pd.read_csv('train.txt', sep=" ", header= None, dtype='float')
df.columns=['X', 'Y', 'Class']
print('\nTrain data')
print(df)
df_arr = df.values
length= len(df_arr)
X1=[]
X2=[]
Y1=[]
Y2=[]

for i in range(length):
    if df_arr[i][2]==1:
        X1.append(df_arr[i][0])
        Y1.append(df_arr[i][1])
    else:
        X2.append(df_arr[i][0])
        Y2.append(df_arr[i][1])


plt.scatter(X1, Y1, color = 'red', marker = '*')    #class1 Red marker '*'
plt.scatter(X2, Y2, color = 'black', marker = '^')  #class2 Black marker '^'

#finding class mean 
X1_mean= st.mean(X1)
X2_mean= st.mean(X2)
Y1_mean= st.mean(Y1)
Y2_mean= st.mean(Y2)

plt.scatter(X1_mean, Y1_mean, color = 'red', marker = 'x') #class1 marker 'x'
plt.scatter(X2_mean, Y2_mean, color = 'black', marker = 'x') #class2 marker 'x'

w1= np.array([[X1_mean], [Y1_mean]])  # 2X1
w1T=w1.T   # 1X2
print(w1)
print(w1T)

w2= np.array([[X2_mean], [Y2_mean]])  # 2X1
w2T=w2.T      # 1X2
print(w2)
print(w2T)

# to read from a text file
#Test data
dftest=pd.read_csv('test.txt', sep=" ", header= None, dtype='float')
dftest.columns=['X', 'Y', 'Class']
print('\nTest data')
print(dftest)

dftest_arr = dftest.values
lengthtest= len(dftest_arr)

new_x=[]
new_y=[]
new_z=[]

g1=[]
g2=[]

trc1=0
trc2=0

for i in range(lengthtest):
    new_x.append(dftest_arr[i][0])
    new_y.append(dftest_arr[i][1])
    new_z.append(dftest_arr[i][2])

    new_sample= np.array([new_x, new_y])  # 2X1
    new_sample_T=new_sample.transpose()      # 1X2

    g1=np.dot(new_sample_T, w1) - 0.5 * np.dot(w1T, w1)
    g2=np.dot(new_sample_T, w2) - 0.5 * np.dot(w2T, w2)

for i in range(lengthtest):    
    if g1[i] > g2[i]:
        trc1=trc1+1
        dftest['New Class'] = 1.0
        plt.scatter(new_x[i], new_y[i], label="Test Class-1", color = 'red', marker = 's')
    else:
        trc2=trc2+1
        dftest['New Class'] = 2.0   #first row class2 te pore dekhe sob 2 ashe
        plt.scatter(new_x[i], new_y[i], label="Test Class-2", color = 'black', marker = 's')

#accuracy counting
count_accurate_1=0
count_accurate_2=0
tec1=0
tec2=0
#trc hocche cal kore jeta pelam, r tec hocche test data te jeta deya ase

for i in range(lengthtest):
    if dftest_arr[i][2]==1:
        tec1=tec1+1
    else:
        tec2=tec2+1
       
print(trc1)
print(trc2)        
print(tec1)
print(tec2)

ac1=0
ac2=0

if trc1==tec1:
    ac1=ac1+1
elif trc2==tec2:
    ac2=ac2+1
elif tec1< trc1:
    tec1=trc1-tec1
    ac1=tec1
elif tec2< trc2:
    tec2=trc2-tec2
    ac2=tec2    
elif tec1> trc1:
    ac1=trc1
elif tec2> trc2:
    ac2=trc2

ac=ac1+ac2        
print(7-ac)
print("Accuracy: ",((7-ac)/7)*100)
#trc= training data count
#tec= test data count

#class-1 accuracy
if tec1 > trc1:
    count_accurate_1= (trc1 / tec1)*100.0
elif tec1 < trc1:
    count_accurate_1= (tec1 / trc1)*100.0
else:
    count_accurate_1=100.0

#class-2 accuracy
if tec2 > trc2:
    count_accurate_2= (trc2 / tec2)*100.0
elif tec2 < trc2:
    count_accurate_2= (tec2 / trc2)*100.0
else:
    count_accurate_2=100.0
  
accuracy= (count_accurate_1 + count_accurate_2) / 2.0
#count diye accuracy ber korar method
print("\n\tAccuracy Method- Part-1")
print("Accuracy for Class-1: ",count_accurate_1, " %")
print("Accuracy for Class-2: ",count_accurate_2, " %")
print("The Average Accuracy of this K-means Classifier is: ",accuracy,"%\n")


#new class diye accuracy ber korar method
"""
print(dftest) 
c=0
if new_z== dftest['New Class']:
    c=c+1
accuracy_ =( c / lengthtest) * 100.0

print("\n\tAccuracy Method- Part-2")
print("The Accuracy of this K-means Classifier is: ",accuracy_,"%\n")
"""

#decision boundary starts
coeff= w1T - w2T
print("Coefficient: ",coeff)

#const= -0.5 * np.linalg.det( np.dot(w1T, w1) - np.dot(w2T, w2) )
#const= -0.5 * (np.dot(w1T, w1) - np.dot(w2T, w2) )
const=0.5
print("Constant: ",const)

x = np.array(np.arange(-4.0,1.0,8.0))
x_len= len(x)
y=[]
for i in range(x_len):
    y.append(  -( np.dot(coeff[i][0] , x[i]) + const )/ coeff[i][1] )
print("x1 = ",x)
print("y1 = ",y)

x_ = np.negative(x)  
y_ = np.negative(y)

print("x2 = ",x_)
print("y2 = ",y_)

line_X=np.array([x,x_])
line_Y=np.array([y,y_])

plt.plot(line_X,line_Y,'--b')

"""
x=[]
y=[]
for i in range(length):    
    x.append(df_arr[i][0])  
    y.append(df_arr[i][1])  

XY= np.array([x, y])
f= np.dot(coeff, XY) + const
plt.plot(f, 'r-')
"""

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Designing a minimum distance to class mean classifier')
plt.gca().legend(('DB','Train C-1','Train C-2', 'Mean C-1', 'Mean C-2','Test C-1', '','Test C-2'))
plt.show()            
 
        


