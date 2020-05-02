# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


# Importing the training set
dataset_train = pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv")
dataset_train = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/monthly-milk-production-pounds.csv")
training_set = dataset_train.iloc[:, 1].values.reshape(-1,1)
dataset_train.head()

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
train_window = 20
for i in range(train_window, 168):
    X_train.append(training_set_scaled[i-train_window:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers)
        self.hidden_cell = (torch.zeros(self.num_layers,input_size,self.hidden_layer_size),
                            torch.zeros(self.num_layers,input_size,self.hidden_layer_size))
        self.j = 0
    def forward(self, input_seq):
        self.j = self.j+1
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq),-1))
        return predictions[-1]
        
 
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

epochs = 50

for i in range(epochs):
    for x, y in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(model.num_layers,model.input_size,model.hidden_layer_size),
                            torch.zeros(model.num_layers,model.input_size,model.hidden_layer_size))
        
        # print(model.j)
        y_pred = model(x)
        # print(model.j)
        
        single_loss = loss_function(y_pred,y)
        single_loss.backward()
        optimizer.step()
        
    if i%2 == 1:
        print(f'Epoch: {i:3} loss: {single_loss.item():10.8f}')

       

model.eval()
pred = []
for i in range(200):
    if i == 0:
        X_test = X_train[-1,:,0].reshape(-1)
        X_test = torch.Tensor(X_test)
    else:
        X_test = torch.cat((X_test[1:],model(X_test)))
    y_pred = model(X_test)
    pred.append(y_pred)
pred = np.array(pred)

plt.plot(np.arange(len(training_set_scaled)),training_set_scaled)
plt.plot(np.arange(len(pred))+len(training_set_scaled),pred)
plt.show()