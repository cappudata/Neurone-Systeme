import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import  pandas as pd


print("data")

bank_data = pd.read_csv("bank_train_data.csv",sep=",")
bank_label = pd.read_csv("bank_train_labels.csv")

bank_data = pd.get_dummies(bank_data)
data = pd.DataFrame.join(bank_data, bank_label)
print(data.shape)

data = data.values          #transformer dataframe sous from un numpy array

#chargement des données
X = data[:,:51]
Y = data[:,-1:]

print("X:", X.shape)

N = X.shape[0]
d = X.shape[1]

#Création d'un vecteur de poids de taille d+1
weights = np.random.randn(d+1,1)
print("shape",weights.shape)

#Ajout d'une première colonne de "1" à la matrice X des entrées
X = np.hstack((np.ones((N,1)),X))
print("new X:", X.shape)

# Calcul du produit scalaire entre la première entrée du dataset et le vecteur de poids
first_entry = X[0,:]
print("shape f:", first_entry.shape)


# Fonction sigmoid
def sigmoid(z):
    return 1 / (1+np.exp(-z))

#Fonction qui calcule la sortie du modèle pour une matrice de points en entrée
#Version naive avec une boucle
def output(X,weights):
    out = np.zeros((N,1))
    for i in range(N):
        out[i] = sigmoid((weights * X[i,:]).sum())
    return out

print(output(X,weights))

#Version avec un calcul matriciel
def output(X,weights):
    return sigmoid(np.dot(X,weights))

#fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):
    return f.round()

#Fonction qui calcule le taux d'erreur en comparant le y prédit avec le y réel
def error_rate(y_pred,y):
    return (y_pred!=y).mean()

#Calcul de la binary cross entropy entre le vecteur de sortie du modèle et le vecteur des targets.
def binary_cross_entropy(f,y):
    return - (y*np.log(f)+ (1-y)*np.log(1-f)).mean()

# Calcul du gradient de l'erreur par rapport aux paramètres du modèle
#a) Version avec une boucle
def gradient(f,y,X):
    grad = np.zeros((d+1,1))

    for j in range(0,d+1):
        grad[j] = -((y-f)*X[:,j]).mean()
        print(-((y-f)*X[:,j]).mean())

    return grad

#Version avec un calcul matriciel
def gradient_dot(f,y,X):
    grad = -np.dot(np.transpose(X),(y-f))/X.shape[0]
    return grad

#Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)

indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]

X_train = X[training_idx,:]
y_train = Y[training_idx]

X_test = X[test_idx,:]
y_test = Y[test_idx]

#Taux d'apprentissage (learning rate)
eta = 0.01

#Apprentissage du modèle et calcul de la performance tous les 100 itérations
nb_epochs = 10000

for i in range(nb_epochs):
    f_train = output(X_train,weights)
    y_pred_train = prediction(f_train)

    grad = gradient_dot(f_train,y_train,X_train)
    weights = weights - eta*grad

    if(i%100==0):
        error_train = error_rate(y_pred_train,y_train)
        loss = binary_cross_entropy(f_train,y_pred_train)

        f_test = output(X_test, weights)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)
        print("iter : " + str(i) +  " error train : " + str(error_train) + " loss " + str(loss) + " error test : " + str(error_test))

# Affichage des paramètres appris du modèle
print("weights")
print(weights)


#Lancement du modèle avec la librairie scikit-learn et affichage des résultats
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred_test = model.predict(X_test)

error_test = error_rate(y_pred_test, y_test)

print("error_test")
print(error_test)

print("Valeurs des poids du model avec scikitlearn")
print(model.intercept_, model.coef_)


#phase test
print("phase test")
#vu que shape de X est (1002,51) donc on doit ajouter une colonne de "1"

bank_test_data = pd.read_csv("public_data/bank_test_data.csv",sep=",")
bank_test_data = pd.get_dummies(bank_test_data)

data = bank_test_data.values

new_X = data[:]                                      #shape (1002,51)
new_X = np.hstack((np.ones((1002,1)),new_X))         #shape (1002,52)

f_train2 = prediction(output(new_X,weights))

np.savetxt("data2.csv",f_train2, delimiter=";")








