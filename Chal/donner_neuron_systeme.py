import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F


#1
print("Question 1:")

bank_data = pd.read_csv("bank_train_data.csv",sep=",")
bank_label = pd.read_csv("bank_train_labels.csv")
print(bank_data.shape)
#print(bank_label.shape)

#2
print("Question 2:")

bank_data = pd.get_dummies(bank_data)
print(bank_data) #Ya une augmentation


x_data = bank_data.values
y_date = bank_label.values
d = x_data.shape[1]

#3
print("Question 3:")

indices = np.random.permutation(bank_data.shape[0])
training_idx, test_idx = indices[:int(bank_data.shape[0]*0.7)], indices[int(bank_data.shape[0]*0.7):]

X_train = x_data[training_idx,:]
Y_train = y_date[training_idx]

X_test = x_data[test_idx,:]
Y_test = y_date[test_idx]

def prediction(f):
    return  f.round()

def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

#creation du modele de régression binaire

class Neural_network_binary_classif(th.nn.Module):
    def __init__(self,d,h1,h2):
        super(Neural_network_binary_classif,self).__init__()

        #les couches du réseau neuronne et ses dimensions
        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, 1)

        #Initialisation les couche
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

        #Pour la sortie
    def forward(self, x):
        phi1 = F.sigmoid(self.layer1(x))
        phi2 = F.sigmoid(self.layer2(phi1))

        return  F.sigmoid(self.layer3(phi2))


nnet = Neural_network_binary_classif(d,200,100)
print(nnet)

device = "cpu"

nnet = nnet.to(device)

X_train = th.from_numpy(X_train).float().to(device)
Y_train = th.from_numpy(Y_train).float().to(device)

X_test = th.from_numpy(X_test).float().to(device)
Y_test = th.from_numpy(Y_test).float().to(device)

eta = 0.01

criterion = th.nn.BCELoss()

optimizer = optim.SGD(nnet.parameters(), lr=eta)

nb_epochs = 100000
pbar = tqdm(range(nb_epochs))

for i in pbar:
    optimizer.zero_grad()

    f_train = nnet(X_train)
    loss = criterion(f_train, Y_train)
    loss.backward()
    optimizer.step()

    if (i % 1000 == 0):
        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,Y_train)
        loss = criterion(f_train,Y_train)

        f_test = nnet(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test,Y_test)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())



#Phase de test
print("Phase de test")

bank_test_data = pd.read_csv("public_data/bank_test_data.csv",sep=",")
bank_test_data = pd.get_dummies(bank_test_data)

data = bank_test_data.values

data_th = th.from_numpy(data).float().to(device)

f_train2 = prediction(nnet(data_th))

np.savetxt("data.csv",f_train2.detach(), delimiter=";")

