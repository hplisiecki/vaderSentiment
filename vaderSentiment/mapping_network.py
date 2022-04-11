
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time

class MappingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(MappingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # more linear layers
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

# load
df = pd.read_csv('vaderSentiment/mapping_set.csv')

df['norm_vader'] = ((df['negative'] + 4) / 8)

# predict norm_vader from prediction
X = df['prediction'].values
y = df['norm_vader'].values

# split into train and test randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# convert to tensors
X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# create network
net = MappingNetwork(1, 1000, 1).to(device)
a = torch.from_numpy(df.prediction.values).float().to(device).view(-1,1)
from pytorch_forecasting.metrics import QuantileLoss
criterion = nn.MSELoss()

from torch.optim.optimizer import Optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4)
from torch.optim.lr_scheduler import ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, 'min')
best_loss = 10
for epoch in range(1000000000):
    net.train()
    # forward pass
    y_pred = net(X_train.view(-1, 1))
    loss = criterion(y_pred, y_train)
    # backward pass

    optimizer.zero_grad()
    loss.backward()
    # scheduler.step(loss)
    optimizer.step()
    # print loss
    if epoch % 10 == 0:
        print(epoch, loss.item())
    # save best
    if loss.item() < 0.01 and loss.item() < best_loss:
        best_loss = loss.item()
        # delete
        torch.save(net.state_dict(), 'vaderSentiment/mapping_network.pt')
    if epoch % 1000 == 0:
        net.eval()
        b = [i[0] for i in net(a).detach().cpu().numpy()]
        fig = pd.to_numeric(pd.Series(b)).hist().figure
        # save
        fig.savefig(f'vaderSentiment/graphs/mapping_network{epoch}.png')
        # close
        plt.clf()



# load from saved
net.load_state_dict(torch.load('vaderSentiment/mapping_network.pt'))
# test
y_pred = net(X_test.view(-1, 1))
# compute loss
loss = criterion(y_pred, y_test)

# save model
torch.save(net.state_dict(), 'vaderSentiment/mapping_network.pt')

b = net(a).detach().cpu().numpy()



plt.figure(0)

