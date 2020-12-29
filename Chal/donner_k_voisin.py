import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F

#charger les datas
print("data")

bank_data = pd.read_csv("bank_train_data.csv",sep=",")
bank_label = pd.read_csv("bank_train_labels.csv")

bank_data = pd.get_dummies(bank_data)
data = pd.DataFrame.join(bank_data, bank_label)

#print(data)
#la function pour calculer la distance entre 2 vecteurs

def euclidian_distance(v1,v2):
    return math.sqrt(((v1-v2)**2).sum())

#Afficher la distance entre le 1er individu et un second individu choisi au hasard dans le dataframe.

X_data = bank_data[:]
X_data = X_data.dropna()
X_data = X_data[:]
Y_data = data["has suscribed"]

print("Distance :")
print(euclidian_distance(X_data.iloc[0], X_data.iloc[1]))

#Function neighbors pour renvoyer les K plus proche de test
#methode TP

"""""
def neighbors (dataframe, test, k):
    new_list = []
    for i in range(dataframe.shape[0]):
        distance = euclidian_distance(test, dataframe.iloc[i,:2])
        new_list.append(distance)

    dataframe["Distance"] = new_list
    data = dataframe.sort_values(by="Distance")

    return data.head(k)
"""

#methode TD
def neighbors(X_data, y_label, x_test, k):
    list_distances = []
    for i in range(X_data.shape[0]):
        distance = euclidian_distance(X_data.iloc[i], x_test)

        list_distances.append(distance)

    df = pd.DataFrame()

    df["label"] = y_label

    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k, :]

print("test neighbors : ")

k = 3
voisin = neighbors(X_data,Y_data,X_data.iloc[10],k)
print(voisin)

#Function Prediction qui renvoie si l'individu est diabetique ou non
def prediction(neighbors):
    mean = neighbors["label"].mean()
    if (mean < 0.5):
        return 0
    else:
        return 1

#je voulais faire cette function prediction (avec 3 paramètres) comme celle j'ai fait en TP3 mais je n'arrive pas

#test
print("test prediction")
pred = prediction(voisin)
print("pred " + str(pred))

#Partage le dataframe : 70% et 30%
print("partage les donnees : ")
#il y 2 facons faire

"""   
indices = np.random.permutation(bank_data.shape[0])
training_idx, test_idx = indices[:int(bank_data.shape[0]*0.7)], indices[int(bank_data.shape[0]*0.7):]

X_train = x_data[training_idx,:]
Y_train = y_date[training_idx]

X_test = x_data[test_idx,:]
Y_test = y_date[test_idx]

"""
data_train = bank_data.sample(frac=0.7,random_state=90)         #X pour 70%
data_label = bank_label.sample(frac=0.7,random_state=90)        #Y pour 70%

data_train_test = bank_data.drop(data_train.index)              #X pour 30%
data_label_test = bank_label.drop(data_label.index)             #Y pour 30%

#renomer et nettoyer les données X et Y pour 70%
X_train = data_train[:]
X_train = X_train.dropna()                       #remove all missing values
X_train = X_train[:]
Y_train = data_label["has suscribed"]

#renomer et nettoyer les données X et Y pour 30%
X_test = data_train_test[:]
X_test = X_test.dropna()
X_test = X_test[:]
Y_test = data_label_test["has suscribed"]

K = 20
#Function taux_error calcule le taux error entre le test et les vrais valeurs

def taux_error():
    taux = 0
    for i in range(X_test.shape[0]):
        pred = neighbors(X_train,Y_train, X_test.iloc[i],k)
        if (prediction(pred) != Y_test.iloc[i]):
            taux = taux + 1
    resul = (taux / Y_test.shape[0])*100
    return  resul

#test
print(taux_error())

print("Nouveau test")

#Phase de test
#pour chaque nouvelles entrée X des données de test, il faut calculer les K plus proches voisin dans X_train
#parmi les résultats des k plus proches voisin de X_train on a les label de Y_train
#On va prédire dans quelle classe cette nouvelle valeurs appartient dans ces plus proches voisin


bank_test_data = pd.read_csv("public_data/bank_test_data.csv",delimiter=",")
bank_test_data = pd.get_dummies(bank_test_data)

X_test1 = bank_test_data[:]
X_test1 = X_test1.dropna()
X_test1 = X_test1[:]

#je teste avec 50 valeurs et ca marche
liste = []
""" 
for i in range(50):
    liste.append(prediction(neighbors(X_train,Y_train,X_test.iloc[i],5)))
print(liste)
"""

#le vrai test de new valeurs.
#jai pas pu enregistré dans un ficher csv car il a mit beaucoup de temps d'éxécuter

for i in range(X_test1.shape[0]):
    liste.append(prediction(neighbors(X_train,Y_train,X_test.iloc[i],5)))
print(liste)

np.savetxt("data3.csv",liste,delimiter=",")