import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
import numpy as np


class SVM():
    def __init__(self, kernel, C=0.5):
        self.kernel = kernel
        self.C = 1

    def fit(self, X, y, verbose = 1):
        self.X = X
        self.y = y
        self.w = [np.ones(self.X.shape[1]+1)/2]

        losses = []
        losses.append(self.cost_function(X,y))

        learning_rate = 0.001
        i = 0
        while losses[0]/self.cost_function(X,y)<1000 and i<=4000:
            w_grad = self.cost_gradient(X,y)
            new_w = self.w[-1]-learning_rate*w_grad
            self.w.append(new_w)
            
            loss = self.cost_function(X,y)
            losses.append(loss)
            if verbose==1 and i%1000==0:
                print("Iteration: ", i, " Losses: ", loss)
                plot_decision_boundary(X,y,self)
                plt.show()
            i+=1
    
    def predict(self, X):
        X = np.concatenate([X,np.ones_like(X)],axis=1)[:,:-1]
        return np.sign(self.w[-1] @ X.T)
    
    def decision_function(self, X):
        X = np.concatenate([X,np.ones_like(X)],axis=1)[:,:-1]
        return (self.w[-1] @ X.T)

    def cost_function(self, X, y):
        
        # Extend dimension for b value
        X = np.concatenate([X,np.ones_like(X)],axis=1)[:,:-1]

        cost = np.mean(0.5*np.linalg.norm(self.w[-1])+
                        self.C*np.maximum(0,1-y*np.dot(self.w[-1],X.T)))
        return cost

    def cost_gradient(self, X, y):
        gradient = 0
        X = np.concatenate([X,np.ones_like(X)],axis=1)[:,:-1]

        for i in range(len(X)):
            if np.maximum(0, 1-y[i]*np.dot(self.w[-1],X[i,:]))==0:
                res = self.w[-1]
            else:
                res = self.w[-1]-self.C*y[i]*X[i,:]
            gradient += res
        gradient = gradient/len(y)
        return gradient

def plot_decision_boundary(X, y, model):
    w = model.w[-1]

    x_max = X.max(axis=0)
    x_min= X.min(axis=0)
    xs = np.array([x_min[0], x_max[0]])

    y_pred = model.predict(X)

    plt.scatter(X[:,0],X[:,1],c=y_pred)
    plt.plot(xs, xs*w[0]/(-w[1])+w[-1]/(-w[1]))
    plt.plot(xs, xs*w[0]/(-w[1])+(w[-1])/(-w[1]), c='r')
    plt.plot(xs, xs*w[0]/(-w[1])+(w[-1])/(-w[1]), c='r')
    plt.ylim([x_min[1], x_max[1]])
    plt.xlim([x_min[0], x_max[0]])

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    # if plot_support:
    #     ax.scatter(model.support_vectors_[:, 0],
    #                model.support_vectors_[:, 1],
    #                s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

if __name__ == "__main__":
    
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=3)
    # X, y = make_classification(n_samples=200, n_features=2, random_state=1)
    # * Centralize the Data
    X-=X.mean(axis=0)


    svm = SVM('linear')
    svm.fit(X,y)
    # print(svm.cost_function(X,y))

    plt.scatter(X[:,0], X[:,1], c=y)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    # plot_decision_boundary(svm)

    # plt.scatter(X[:,0], X[:,1],c=y)