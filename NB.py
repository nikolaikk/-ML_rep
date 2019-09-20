# FROM https://www.python-course.eu/naive_bayes_classifier_introduction.php
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np
plt.style.use("ggplot")
#%matplotlib inline


## PART TWO ##
df = pd.read_csv("https://www.python-course.eu/data/person_data.txt", header=None, 
                 names = ["Name","SecondN","Height","Weight","Gender"], sep = " ")

num_b = int((np.max(df["Height"])- np.min(df["Height"]))/5)

#plt.hist(df[df["Gender"]=="male"]["Height"],bins = num_b, label="male", alpha = 0.5,rwidth = 0.75)
#plt.hist(df[df["Gender"]=="female"]["Height"],bins = num_b, label="female", alpha = 0.5,rwidth=0.75)
#plt.legend()
#plt.show()

#plt.figure()
#y, _ = pd.factorize(df["Gender"])
#plt.scatter(df["Height"],df["Weight"],c=y)


train = df.iloc[:-20,:]
test = df.iloc[-20:,:]
y, _ = pd.factorize(train["Gender"])
#plt.scatter(train["Height"], train["Weight"], c=y)

def posterior_max(X, data = train):
    
    
    # Male conditional probability
    def conditional (x, y, data = train):
        """ Computes conditional probability"""
        prob = []
        features = train[train["Gender"]==y].iloc[:,2:4]
        cov = np.cov(features.T)
        mean = np.mean(features)
#        X_train = train.iloc[:,2:4]
        
        for i in range(x.shape[0]):
            exp_ = np.exp(np.dot(-0.5*(x.iloc[i,:]-mean).dot(np.linalg.inv(cov)),
                                 (x.iloc[i,:]-mean).T))
            
#            print(x.shape[1])
            prob.append(exp_/np.sqrt((4*np.pi)**x.shape[1]*np.linalg.det(cov)))
        
        return np.array(prob).reshape(-1,1)
    
    def prior(y):
        count_y = data[data["Gender"]==y].shape[0]
        count_total = data.groupby("Gender").size().sum()
        return count_y/count_total
    
    def posterior(X, y_):
        result = prior(y_)*conditional(X,y_)/(prior("male")*conditional(X,"male")+
                   prior("female")*conditional(X,"female"))
        return result
        
    
    max_ = []
    for i in ["male","female"]:
        res = posterior(X, i)
        print(f"P({i}|x = [Height:{np.array(X)[0][0]}, Weight:{np.array(X)[0][1]}]) = {res}")
        max_.append(res)
    
    if max_[0] > max_[1]:
        print("male")
        return "male"
    else:
        print("female")
        return "female"


# Print classes for all from test\
y_pred = []

for i in range(test.shape[0]):
    prediction = posterior_max(test.iloc[i:i+1,2:4])
    y_pred.append(prediction)
        
        
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred,list(test.iloc[:,-1])))
    

# Sklearn
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train.iloc[:,2:4],train.iloc[:,-1])
predicted = model.predict(test.iloc[:,2:4])
print(confusion_matrix(y_pred,list(test.iloc[:,-1])))



#####################################3



#def prior(y, data=train):
#        count_y = data[data["Gender"]==y].shape[0]
#        count_total = data.groupby("Gender").size().sum()
#        return count_y/count_total
#
#class1 = train[train["Gender"]=="male"].iloc[:,2:4]
#class2 = train[train["Gender"]=="female"].iloc[:,2:4]
#cov1 = np.cov(class1.T)
#cov2 = np.cov(class2.T)
#mean1 = np.mean(np.array(class1), axis=0)
#mean2 = np.mean(np.array(class2), axis=0)
#
#A = 0.5*(np.linalg.inv(cov1) - np.linalg.inv(cov2))
#alpha = np.dot(mean1,np.linalg.inv(cov1)) - np.dot(mean2,np.linalg.inv(cov2))   
#alpha0 = np.log(prior("male")/prior("female")) + \
#        1/2*(np.log(np.linalg.det(2*np.pi*cov2)/np.linalg.det(2*np.pi*cov1)) + \
#        np.dot(np.dot(mean1,np.linalg.inv(cov2)),mean1.T) - \
#        np.dot(np.dot(mean1,np.linalg.inv(cov1)),mean1.T))
#        
#def solve_quadratic_eq(a,b,c):
#    disc = np.sqrt(b**2-4*a*c)
#    return (-b+disc)/(2*a), (-b-disc)/(2*a)
#
#x = np.linspace(120,220,300)
#
#output1 = np.empty(len(x_input))
#output2 = np.empty(len(x_input))
#for i in range(len(x_input)):
#    a = A[1,1]
#    b = (A[1,0] + A[0,1])*x[i] + alpha[1]
#    c = A[0,0] * x[i]**2 + alpha[0] * x[i] + alpha0 
#    output1[i], output2[i] = solve_quadratic_eq(a,b,c)
#    
#plt.figure()
#plt.scatter(train["Height"], train["Weight"], c=y)
#plt.plot(x,output1, label = "first sol")
#plt.plot(x,output2, label = "second sol")
#plt.legend()



#########################################3
X, y = make_blobs(300,2,centers=2, random_state=100)

X[y==0] = X[y==0]+2
X[y==0] = np.dot(np.array([[0.5,0.1],[1.2,1.5]]),X[y==0].T).T
X = X - np.mean(X,axis=0)
class1 = X[y==0]
class2 = X[y==1]
prior1 = len(class1)/len(X)
prior2 = len(class2)/len(X)
cov1 = np.cov(class1.T)
cov2 = np.cov(class2.T)
#cov1[1,0] = cov1[0,1] = cov2[0,1] = cov2[1,0] = 0
mean1 = np.mean(np.array(class1), axis=0)
mean2 = np.mean(np.array(class2), axis=0)

A = 0.5*(np.linalg.inv(cov1) - np.linalg.inv(cov2))
alpha = np.dot(mean1,np.linalg.inv(cov1)) - np.dot(mean2,np.linalg.inv(cov2))   
alpha0 = np.log10(prior1/prior2) + \
        0.5*(np.log10(np.linalg.det(2*np.pi*cov2)/np.linalg.det(2*np.pi*cov1)) + \
        np.dot(np.dot(mean1,np.linalg.inv(cov2)),mean1.T) - \
        np.dot(np.dot(mean1,np.linalg.inv(cov1)),mean1.T))
        
def solve_quadratic_eq(a,b,c):
    disc = np.sqrt(b**2-4*a*c)
    return (-b+disc)/(2*a), (-b-disc)/(2*a)

x = np.linspace(np.min(X[:,0]),np.max(X[:,0]),1000)

output1 = np.empty(len(x))
output2 = np.empty(len(x))
for i in range(len(x)):
    a = A[1,1]
    b = (A[1,0] + A[0,1])*x[i] + alpha[1]
    c = A[0,0] * x[i]**2 + alpha[0] * x[i] + alpha0 
    output1[i], output2[i] = solve_quadratic_eq(a,b,c)
    
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(x,output1, label = "first sol")
plt.plot(x,output2, label = "second sol")
plt.legend()