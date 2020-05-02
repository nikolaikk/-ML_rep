import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
import matplotlib 
# matplotlib.use('')


class NeuralNet():
    def __init__(self, input_size, output_size, layers, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.weights = self.create_weight_bias()
        self.losses = []
        self.activation = activation
        
    def create_weight_bias(self, random=True, name='', factor=1):
        # np.random.seed(42)
        weights = {}
        layers = [self.input_size, *self.layers, self.output_size]
        for i, (layer1, layer2) in enumerate(zip(layers[:-1],layers[1:])):
            
            weights[name+f"W{i+1}"] = np.random.rand(layer1,layer2) if random \
                                    else np.zeros((layer1,layer2))*factor
            weights[name+f"b{i+1}"] = np.ones([layer2,1]) if random \
                                    else np.ones((layer2,1))*factor
                                    
        return weights
            
    def sigma(self,Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0,Z)

    def sigma_prime(self, Z):
        return self.sigma(Z) * (1 - self.sigma(Z))
    
    def relu_prime(self, Z):
        return np.heaviside(Z,1)
    
    def softmax(self, Z):
        Z -= np.max(Z)
        return np.exp(Z)/np.exp(Z).sum(1,keepdims=True)
    
    def softmax_prime(self, Z):
            return 1/np.sum(np.exp(Z),1,keepdims=True)**2*np.exp(Z)\
                *(np.sum(np.exp(Z),1,keepdims=True)-np.exp(Z))

    def loss_fn(self, y, y_prime, kind='CrossEntropy'):
        assert kind in ['CrossEntropy', 'MSE'], 'Loss functiion is not supported!'+\
            '\n Avalilable loss funcitons ["MSE", "CrossEntropy"]'
            
        self.y = y
        self.loss_function = kind
            
        if self.loss_function == 'CrossEntropy':
            loss = -(y*np.log(y_prime)).sum(1).mean()
            self.losses.append(loss)
        
        if self.loss_function == 'MSE':
            loss = ((y-y_prime)**2).sum(axis=1).mean()
            self.losses.append(loss)
        
        return loss

        
    def loss_fn_prime(self, y, y_prime):
        assert self.loss_function in ['CrossEntropy', 'MSE'], 'Loss functiion is not supported!'+\
            '\n Avalilable loss funcitons ["MSE", "CrossEntropy"]'
            
        if self.loss_function == 'CrossEntropy':
            # loss = -(y/(y_prime))
            loss = -(y/(y_prime)-(1-y)/(1-y_prime))
            self.losses.append(loss)
        
        if self.loss_function == 'MSE':
            # TODO maybe remove sum method and mean(axis=0)
            loss = 2*((self.y-y_prime)).sum(axis=1).mean()
            self.losses.append(loss)
        
        return loss
    
    
    def forward(self, X):
        
        self.data = X
        if self.activation.lower() == 'relu':
            activation_fn = self.relu
        if self.activation.lower() == 'sigma':
            activation_fn = self.sigma
            
        z_s, a_s = [], [X]
        
        Ws = list(net.weights.values())[::2]
        bs = list(net.weights.values())[1::2]
        for i, (W, b) in enumerate(zip(Ws,bs)):

            if i==len(Ws)-1:
                activation_fn = self.softmax
 
            z = (X.dot(W).T + b).T
            z_s.append(z)

            X = activation_fn(z.copy())
            a_s.append(X)
            
        self.a_s = a_s
        self.z_s = z_s
        return X


    def BackPropagation(self, alpha=0.01, update_param=True):
        '''
        Calculates back propagation
        dE/dw = dE/da * da/dz * dz/dw
        '''
        delta_weights = self.create_weight_bias(random=False,name='delta_',factor=0)
        self.alpha = alpha

        y_pred = self.forward(self.data)
        
        for i in reversed(range(len(self.layers)+1)):
            # Back propagate gradient of loss function
            if i == len(self.layers):
                dEda = self.loss_fn_prime(self.y, y_pred)
                dadz = self.softmax_prime(self.z_s[i])
                delta = dEda * dadz


            else:
                # W = self.weights[list(self.weights.keys())[i*2]]
                W = self.weights[list(self.weights.keys())[2*i+2]]
                
                if self.activation == 'relu':
                    activation_fn = self.relu_prime
                if self.activation == 'sigma':
                    activation_fn = self.sigma_prime
                    
                dadz = activation_fn(self.z_s[i].copy())
                delta = np.dot(W, delta.T).T*dadz

            dzdw = self.a_s[i]
                        
            deltab = delta.mean(0,keepdims=True).T#[np.newaxis].T
            deltaW = np.dot(dzdw.T, delta)/self.data.shape[0]
              
            assert (list(delta_weights.values())[2*i].shape == deltaW.shape
                    and list(delta_weights.values())[2*i+1].shape == deltab.shape),\
                'Error. Wrong shape!'
                                    
            delta_weights[list(delta_weights.keys())[i*2]] = deltaW
            delta_weights[list(delta_weights.keys())[i*2+1]] = deltab
                                    
        self.delta_weights = delta_weights

        # Update weights
        if update_param:
            for old_param, delta_param in zip(self.weights.keys(), delta_weights.keys()):
                self.weights[old_param] += -self.alpha * delta_weights[delta_param]
            
    
    def __call__(self,X):
        return self.forward(X)
    
    
def plot_with_contours(X, y_pred, model):
    
    # plot countour
    points = 300
    x_ = np.linspace(X[:,0].min()-1,X[:,0].max()+1,points)
    y_ = np.linspace(X[:,1].min()-1,X[:,1].max()+1,points)
    
    X_, Y_ = np.meshgrid(x_, y_)
    
    Z_ = model.forward(np.stack([X_.flatten(),Y_.flatten()]).T)[:,1]
    Z_ = Z_.reshape(points,points)
    
    # Decision boundary
    # Z = Z_>0.5    
    # z = Z[:,:-1]*(~Z[:,1:])

    plt.close()
    plt.figure()    
    # cp = plt.contourf(X_, Y_, Z_)
    cp = plt.pcolormesh(X_, Y_, Z_, cmap='coolwarm')
    plt.colorbar(cp)
    plt.scatter(X[:,0], X[:,1], c=y_pred, edgecolors='red')
    plt.contour(X_, Y_, Z_, levels=[0.499,0.501],cmap='gist_heat')
    plt.show()
    
    return Z_

def OneHotEncoder(y, classes=None):
    assert len(y.shape)==1, 'It is not one dimensional data'
    if classes is None:
        classes = np.unique(y)
    y_encoded = np.zeros([len(y),classes])
    for i, val in enumerate(y):
        y_encoded[i,val]=1
    return y_encoded
    
def create_test_weight(input_size=3,layers=[3],output_size=3):  
    pass 
    
if __name__=="__main__":
    def normalize_centralize(X):
        X -= X.mean(0)
        X /= np.abs(X).max(0)
        return X


    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=3)
    y = OneHotEncoder(y,2)

    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)    
    y = OneHotEncoder(y,2)
    
    # X = np.array([[0.1,0.7],[0.6,0.8],[0.7,0.7]])
    # y = np.array([[1,0],[0,1],[0,1]])


    # X = normalize_centralize(X)
    net = NeuralNet(2,2,[50,3],activation='sigma')
    net.create_weight_bias(random=False)

    for i in range(40000):
        y_pred = net(X)
        # loss = net.loss_fn(y, y_pred, kind='MSE')
        loss = net.loss_fn(y, y_pred, kind='CrossEntropy')
        net.BackPropagation(alpha=0.02, update_param=True)
        if i%1000==0:
            y_pred = net.forward(X)
            
            print("Iteration", i, "Loss: ", loss, "   Accuracy: "
                    ,accuracy_score(y.argmax(1), y_pred.argmax(1))
                    )
    
    z_ = plot_with_contours(X,y_pred.argmax(1), net)

    