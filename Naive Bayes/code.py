"""Anika Salsabil
  AUST, CSE
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy.integrate import odeint
#import sympy as sym

#Test data
df=pd.read_csv('test.txt', sep=",", header= None, dtype='float')
print("\nTest data\n", df)
df_arr = df.values
length= len(df_arr)

#Given data
u1=np.array([[0, 0]])
E1=np.array([[0.25, 0.3], [0.3, 1]])
u2=np.array([[2, 2]])
E2=np.array([[0.5, 0], [0, 0.5]])

Pw1, Pw2= 0.5, 0.5
legend1, legend2=0,0
list_N1, list_N2, list_Pc1, list_Pc2, x, y=[],[],[],[],[],[]

dby=[]

"""
######TASK-1
"""
for row in df_arr:
    Xi=np.array(row)
    x.append(Xi[0])
    y.append(Xi[1])
    
    x_u1= Xi - u1     # xi- u1
    x_u1_T= x_u1.transpose()   # (xi- u1) er transpose
    E1_inv= np.linalg.inv(E1) 
        
    x_u2= Xi - u2     # xi- u2
    x_u2_T= x_u2.transpose()   # (xi- u2) er transpose
    E2_inv= np.linalg.inv(E2) 
    
    detE1=np.linalg.det(E1)
    detE2=np.linalg.det(E2)
    
    twopi= 2* math.pi
    twopi_d= np.power(twopi,2)  #for d=2 (dimension)
    d=2
    N1= -(d/2)*np.log(twopi) -0.5* np.log(detE1) - 0.5* np.dot(np.dot(x_u1, E1_inv) , x_u1_T) 
    N2= -(d/2)*np.log(twopi) -0.5* np.log(detE2) - 0.5* np.dot(np.dot(x_u2, E2_inv) , x_u2_T) 
    
    list_N1.append(N1)
    list_N2.append(N2) 
    
    Pc1= np.exp(N1 + np.log(Pw1))
    Pc2= np.exp(N2 + np.log(Pw2))
    list_Pc1.append(Pc1)
    list_Pc2.append(Pc2)
    
    dby.append(Pc1)
    dby.append(Pc2)
    
    if Pc1 > Pc2:
        plt.scatter(Xi[0], Xi[1], color = 'red', marker = '*', label= 'Test C-1') 
    else:
        plt.scatter(Xi[0], Xi[1], color = 'black', marker = '*', label= 'Test C-2') 

print("\nN1: \n",N1) 
print("\nN2: \n",N2)  
print("\nTrue class probability of Class-1: \n",Pc1) 
print("\nTrue class probability of Class-2: \n",Pc2)       
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Implementing Minimum Error Rate Classifier')

"""
######TASK-2
"""
colors = ['red', 'black']
lines = [Line2D([0], [0], color=c, linewidth=0, marker='*') for c in colors]
labels = ['Test C-1', 'Test C-2']
plt.legend(lines, labels) 
plt.show()

"""
######TASK-3
"""
fig = plt.figure(figsize=(9, 7))
ax = fig.gca(projection='3d')
z=np.zeros(6)
#3d plotting
for i in range(length):
    if list_Pc1[i] > list_Pc2[i]:
        #dby.append((list_Pc1[i] - list_Pc2[i]))
        ax.scatter(x[i], y[i], z[i], color='red', marker='o')  
        
    else:
        #dby.append((list_Pc2[i] - list_Pc1[i]))             ###########
        ax.scatter(x[i], y[i], z[i], color='black', marker='o') 
        
N = 100
X = np.linspace(-6, 6, N)
Y = np.linspace(-6, 6, N)
X, Y = np.meshgrid(X, Y)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, u, E):
    detE = np.linalg.det(E)
    E_inv = np.linalg.inv(E)
    #N = np.sqrt((2*np.pi)**n * detE)
    N=np.sqrt(twopi_d * detE)
    fac = np.einsum('...k,kl,...l->...', pos-u, E_inv, pos-u)
    return np.exp(-fac / 2) / N

Z1 = multivariate_gaussian(pos, u1, E1)
Z2 = multivariate_gaussian(pos, u2, E2)
Z=  (Z1 -Z2)

ax.plot_surface(X, Y, Z1, rstride=3, cstride=3, cmap=cm.viridis, alpha=0.4)
ax.contour(X, Y, Z1, zdir='z', offset=-0.15)
ax.text(0, 0, 0.4, "W1", color='red')

ax.plot_surface(X, Y, Z2, rstride=3, cstride=3, cmap=cm.viridis, alpha=0.4)
ax.contour(X, Y, Z2, zdir='z', offset=-0.15)
ax.text(5, 0.8, 0.3, "W2", color='black')

"""
######TASK-4
"""
# plot DB
ax.contour(X, Y, Z, zdir='z', offset=-0.15)

colors = ['red', 'black']
lines = [Line2D([0], [0], color=c, linewidth=0, marker='o') for c in colors]
labels = ['Test C-1', 'Test C-2']
plt.legend(lines, labels)
ax.set_zlim(-0.15, 0.3)
ax.set_zticks(np.linspace(0, 0.3, 10))
#ax.view_init(27, -21)
ax.view_init(azim=30)
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)               
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Probability Density")
plt.show()





# plot DB
#ax.plot(dby,dbx, [-0.15], zdir='z', color='tomato')


"""
p1= -(d/2)*np.log(twopi) -0.5* np.log(detE1)
p2= -(d/2)*np.log(twopi) -0.5* np.log(detE2)
a=(p1-p2)

b= np.log(Pw2)- np.log(Pw1)
#S=  a + b - 0.5* np.dot(np.dot( (S- u1), E1_inv) , (S- u1).transpose())  +  0.5* np.dot(np.dot( (S- u2), E2_inv) , (S- u2).transpose())
sym.init_printing()
x,y=sym.symbols('x,y')
print("\nEquation solved: \n")
sym.Eq(a + b - 0.5* np.dot(np.dot( (x- u1), E1_inv) , (x- u1).transpose())  +  0.5* np.dot(np.dot( (x- u2), E2_inv) , (x- u2).transpose(), 0)

sym.solve(a + b - 0.5* np.dot(np.dot( (x- u1), E1_inv) , (x- u1).transpose())  +  0.5* np.dot(np.dot( (x- u2), E2_inv) , (x- u2).transpose(), x)
"""







