import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import time


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6, shuffle = False)
X-=np.mean(X,axis=0)
        
# ass = [1, 4, 20,-71,-10,-3,0.1,0.2,-0.05,-0.13]

ass = [1, 2.1, 4,7, 15,71]
ass1 = [1, 2.1, 20, 45,67, 80.1]
ass1 = [1, 2.1, 4,7, 15,71]
bss = [0.45]

c = np.array(["red","blue"])
text_color = ["white","aqua"]
        
x_line = np.arange(-5,6,0.1)

def lin_f(a,b,x):
    return a*x+b

def indic(x1,x2):
    output = np.empty(len(x1))
    assert len(x1) == len(x2), "Wrong length"
    for i in range(len(x1)):
        if x1[i]==x2[i]:
            output[i] = 0
        else:
            output[i] = 1
    
    return output
        


        
    
W = np.zeros(len(X))+1/len(X)
err = []
alpha = []
G_m = []
Ws = []

fig, ax = plt.subplots()
ax.set_xlim(-4,5)
ax.set_ylim(-7.5,7.5)

for i in range(len(ass)):
    # X[:,1]-lin_f(1,-0.8,X[:,0])    
    y_pred = (np.sign(lin_f(ass[i],bss[0],X[:,0])-X[:,1])+1)/2
    
    G_m.append(y_pred)
    
    err_val = np.sum(W*indic(y,y_pred))/np.sum(W)
    err.append(err_val)
    
    alpha_val = np.log((1-err_val)/err_val)
    alpha.append(alpha_val)
    
    W = W * np.exp(alpha_val*indic(y,y_pred))
    Ws.append(W)
    
    if i == 0:
        scatter_ = ax.scatter(X[:,0],X[:,1],c=c[y_pred.astype(int)],s = W*1000)
        plot_, = ax.plot(x_line,lin_f(ass[i],bss[0],x_line),'grey')
        fig.canvas.draw()
        plt.pause(1)
    else:
        scatter_.set_sizes(W*1000)
        scatter_.set_color(c[y_pred.astype(int)])
        ax.plot(x_line, lin_f(ass[i],bss[0],x_line),'grey')
        plt.pause(1)
        fig.canvas.draw()
        
    
        
G = np.sign(np.sum([alpha[i]*G_m[i] for i in range(len(alpha))],axis=0))


# CREATE SUBPLOTS

W = np.zeros(len(X))+1/len(X)
err = []
alpha = []
G_m = []
Ws = []

fig1, ax1 = plt.subplots(2,3)

for i in range(len(ass1)):
    # X[:,1]-lin_f(1,-0.8,X[:,0])    
    y_pred = (np.sign(lin_f(ass1[i],bss[0],X[:,0])-X[:,1])+1)/2
    
    G_m.append(y_pred)
    
    err_val = np.sum(W*indic(y,y_pred))/np.sum(W)
    err.append(err_val)
    
    alpha_val = np.log((1-err_val)/err_val)
    alpha.append(alpha_val)
    
    W = W * np.exp(alpha_val*indic(y,y_pred))
    Ws.append(W)

    
    ax1[i//3,i%3].scatter(X[:,0],X[:,1],c=c[y_pred.astype(int)],s = W*1000)
    for ii in range(len(X)):
        ax1[i//3,i%3].text(X[ii,0],X[ii,1],str(ii), color = text_color[y[ii]])
    
    for j in range(len(alpha)):
        ax1[i//3,i%3].plot(x_line,lin_f(ass1[j],bss[0],x_line),'grey')
    ax1[i//3,i%3].set_xlim(-4,5)
    ax1[i//3,i%3].set_ylim(-7.5,7.5)
    