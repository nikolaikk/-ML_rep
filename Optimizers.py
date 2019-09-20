
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import normalize

import scipy as sp


#xx, yy = make_regression(n_samples = 100, n_features=1, n_informative=1, noise=20, random_state=12)
#xx = normalize(xx.reshape(-1,1),axis=0).ravel()
#yy = normalize(yy.reshape(-1,1),axis=0).ravel()
np.random.seed(100)
xx = 2 * np.random.rand(100,1)
yy = (4 +3 * xx+np.random.randn(100,1)).ravel()
xx = xx.ravel()
xx = xx-np.mean(xx)/np.max(xx-np.mean(xx))
yy = yy-np.mean(yy)/np.max(yy-np.mean(yy))

a = -2
b = 2
rate = 0.001

def error_f(xx,yy, a=a, b=b):
    return np.mean(((a*xx+b)-yy)**2)

def dif_a_error_f(xx,yy, a=a, b=b):
    return np.mean(2*(a*xx+b-yy)*xx)

def dif_b_error_f(xx,yy, a=a, b=b):
    return np.mean(2*(a*xx+b-yy))



def gd(a,b,xx,yy,rate):
    rate = rate*100
    iterations = 0
    a_s, b_s, err = [], [], []
    while error_f(xx,yy,a,b)>1.16 and iterations <50:
        if iterations == 0:
            print(a,b)
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)

        a += -rate*dif_a_error_f(xx,yy,a,b)
        b += -rate*dif_b_error_f(xx,yy,a,b)
    
        iterations+=1
        if iterations%500==0:
            print("Error :", error_f(xx,yy,a_s[-1],b_s[-1]))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err


def sgd(a,b,xx,yy,rate):
    
    rate=rate
    iterations = 0
    a_s, b_s, err = [], [], []
    while error_f(xx,yy,a,b)>1.16 and iterations <50:
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)
        
        for i in range(len(xx)):
            index = np.random.choice(np.arange(len(xx)))
            xx_r = xx[index]
            yy_r = yy[index]
            a += -rate*dif_a_error_f(xx_r,yy_r,a,b)
            b += -rate*dif_b_error_f(xx_r,yy_r,a,b)
    
        iterations+=1
        if iterations%500==0:
            print("Error :", error_f(xx,yy,a,b))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err



def sgd_batch(a,b,xx,yy,rate, batch = 10):

    rate=rate*10
    iterations = 0
    a_s, b_s, err = [], [], []
    while error_f(xx,yy,a,b)>1.16:
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)

        for i in range(int(len(xx)/batch)):
            index = np.random.choice(np.arange(len(xx)), batch)
            xx_r = xx[index]
            yy_r = yy[index]
            a += -rate*dif_a_error_f(xx_r,yy_r,a,b)
            b += -rate*dif_b_error_f(xx_r,yy_r,a,b)
    
        iterations+=1
        if iterations%50==0:
            print("Error :", error_f(xx,yy,a,b))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err

def sgd_batch_momentum(a,b,xx,yy,rate, batch = 10):

    rate=rate*10
    iterations = 0
    a_s, b_s, err = [], [], []
    beta = 0.95
    V_da, V_db = 0, 0
    while error_f(xx,yy,a,b)>1.16:
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)

        for i in range(int(len(xx)/batch)):
            index = np.random.choice(np.arange(len(xx)), batch)
            xx_r = xx[index]
            yy_r = yy[index]
            
            V_da = beta*V_da+(1-beta)*dif_a_error_f(xx_r,yy_r,a,b)
            V_db = beta*V_db+(1-beta)*dif_b_error_f(xx_r,yy_r,a,b)
#            print(V_da, V_db)
            a += -rate*V_da
            b += -rate*V_db
    
        iterations+=1
        if iterations%10==0:
            print("Error :", error_f(xx,yy,a,b))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err

def RMS_prop(a,b,xx,yy,rate, batch = 10):

    rate=rate*10
    iterations = 0
    a_s, b_s, err = [], [], []
    
    beta = 0.90
    V_da, V_db = 0, 0
    epsilon = 1
    while error_f(xx,yy,a,b)>1.16:
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)

        for i in range(int(len(xx)/batch)):
            index = np.random.choice(np.arange(len(xx)), batch)
            xx_r = xx[index]
            yy_r = yy[index]
            
            V_da = beta*V_da+(1-beta)*dif_a_error_f(xx_r,yy_r,a,b)**2
            V_db = beta*V_db+(1-beta)*dif_b_error_f(xx_r,yy_r,a,b)**2
#            print(V_da, V_db)
            a += -rate * dif_a_error_f(xx_r,yy_r,a,b)/(np.sqrt(V_da)+epsilon)
            b += -rate * dif_b_error_f(xx_r,yy_r,a,b)/(np.sqrt(V_db)+epsilon)
    
        iterations+=1
        if iterations%10==0:
            print("Error :", error_f(xx,yy,a,b))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err

def AdaM(a,b,xx,yy,rate, batch = 10):

    rate=rate*10
    iterations = 0
    a_s, b_s, err = [], [], []
    
    beta1 = 0.90
    beta2 = 0.8
    V_da, V_db = 0, 0
    S_da, S_db = 0, 0
    epsilon = 1
    while error_f(xx,yy,a,b)>1.16:
        err.append(error_f(xx,yy,a,b))
        a_s.append(a)
        b_s.append(b)

        for i in range(int(len(xx)/batch)):
            index = np.random.choice(np.arange(len(xx)), batch)
            xx_r = xx[index]
            yy_r = yy[index]
            
            V_da = beta1*V_da+(1-beta1)*dif_a_error_f(xx_r,yy_r,a,b)
            V_db = beta1*V_db+(1-beta1)*dif_b_error_f(xx_r,yy_r,a,b)
            
            V_da_corrected = V_da/(1-beta1)
            V_db_corrected = V_db/(1-beta1)        
            
            S_da = beta2*S_da+(1-beta2)*dif_a_error_f(xx_r,yy_r,a,b)**2
            S_db = beta2*S_db+(1-beta2)*dif_b_error_f(xx_r,yy_r,a,b)**2
            
            S_da_corrected = S_da/(1-beta2)
            S_db_corrected = S_db/(1-beta2) 
            
            
#            print(V_da, V_db)
            a += -rate * V_da_corrected/(np.sqrt(S_da_corrected)+epsilon)
            b += -rate * V_db_corrected/(np.sqrt(S_db_corrected)+epsilon)
    
        iterations+=1
        if iterations%10==0:
            print("Error :", error_f(xx,yy,a,b))
            print("Iterations :", iterations)
    print("Number of iteratioins {}:".format(iterations))
    return a_s, b_s, err
   


#theta1, theta2, err = sgd_batch(a,b,xx,yy,rate)
theta1, theta2, err = sgd_batch_momentum(a,b,xx,yy,rate)
theta1, theta2, err = RMS_prop(a,b,xx,yy,rate)
theta1, theta2, err = AdaM(a,b,xx,yy,rate)




a_s=np.linspace(theta1[-1]-5,theta1[-1]+5,100)
b_s=np.linspace(theta2[-1]-5,theta2[-1]+5,100)
res=np.ones([len(a_s),len(b_s)])
for ii,i in enumerate(a_s):
    for jj, j in enumerate(b_s):
        res[ii,jj] = error_f(xx,yy,i,j)


aa, bb  = np.meshgrid(a_s,b_s, sparse=False, indexing='ij')
fig, ax = plt.subplots(2,1, figsize=(9,15))
im = ax[1].contour(aa,bb,res,50)
#fig.colorbar(im, cax=ax[1])

for i in range(len(theta1)-1):
    ax[1].annotate(' ', xy=(theta1[i+1],theta2[i+1]), xytext=(theta1[i],theta2[i]),
                arrowprops={'arrowstyle': '->'}, va='center')


ax[0].scatter(xx, yy, c='navy', s=50, marker='*', alpha=0.5, label='Real Data')
ax[0].plot(xx, xx*theta1[-1]+theta2[-1],c='darkorange', linewidth=7, label='Gradient Descent')

slope1, intercept1, _,_,_ = sp.stats.linregress(xx,yy)
ax[0].plot(xx, xx*slope1+intercept1, c='black', linewidth=1, label='Least Squares')
ax[0].legend(fontsize=15);