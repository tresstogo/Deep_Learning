## Boltzmann Machine

## Importing Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', 
                     sep = "::", header = None, engine = "python", encoding = "latin-1")
## Se necesitan especificar lso params del csv_read debido 
## a que el archivo que se está leyendo no es un csv_file como tal si no un Dat File
users = pd.read_csv('ml-1m/users.dat', 
                     sep = "::", header = None, engine = "python", encoding = "latin-1")
ratings = pd.read_csv('ml-1m/ratings.dat', 
                     sep = "::", header = None, engine = "python", encoding = "latin-1")

training_set = pd.read_csv("ml-100k/u1.base", delimiter = "\t")
training_set = np.array(training_set, dtype = "int")
test_set = pd.read_csv("ml-100k/u1.test", delimiter = "\t")
test_set = np.array(test_set, dtype = "int")
##Convertimos los sets de Data Frames a Arrays 

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0]))) ## Encontramos los valores máximos
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

## Restricted Boltzmann Machine son un tipo de NN

##Converting the data into an array with users in lines and movies in columns
## Lista de Listas
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
## Creamos Tensors, los tensors son Arrays con un solo tipo de dato

## Converting this data into Torch Tensors (Más eficiente, mismo arreglo pero ahora es un Torch Tensor)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

## Converting the ratings into binary ratings 1(Liked) or 0(Not Liked)
## Convertiremos el trainig y el test en resultados binarios como se explico arriba
## Los resultados no registrados (los 0s) se igualaran a -1 con la siguiente sintáxis
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
## El valor en corchetes es una condición que indica que el elemento dentro de la lista que la cumpla se igualará a -1
## Todo aquel rating que tenga un valor de 1 o 2 se igualará a cero, ya que no le gustó al usuario
## Si un rating tiene 3 estrellas o más, quiere decir que le gustó al usuario por lo que se iguala a 1

## Creating the architecture of the NN
class RBM():
    ## Number of visible and Number of Hidden Nodes
    def __init__(self,nv, nh):
        self.W = torch.rand(nh, nv)
        self.a = torch.rand(1, nh)
        self.b = torch.rand(1, nv)
        ## BIAS for visible and hidden nodes

## La siguiente función será para calcular la probabilidad de las hidden nodes dadas los visibles 
## (sigmoid activation function). Probabilidad de que el node h sea 1 dado v, esta probabilidad será igual 
## a la activation function
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
    ## El sample de h es la sigmoid function respecto al producto de los weights
    ## por el número de neuronas (x) más el BAIS (a)
        activation = wx + self.a.expand_as(wx)
        ## Se pone el expand debido a que se tiene que tomar acorde al batch y aplicar al BIAS a cada línea del batch (???)
        p_h_given_v = torch.sigmoid(activation)
        ## Se puede pensar en un ejemplo en el que "h" representa el género de películas en general 
        ## y "v" el género de la película evaluada en la iteración, por lo que el valor será alto si "v" tiene relación con "h"
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    ## Bernoulli realiza el sampling de los hidden nodes de la siguiente forma:
    ## Suponiendo que un valor en p_h_given_v es 0.7, Bernoulli toma un valor random de 0 al 1 y si es mayor que el valor p_h_given_v
    ## el valor de esa posición de Bernoulli es 0 de lo contrario es 1
    ## Bernoulli sirve para activar o no la neurona.
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        ## mm es el método para multiplicar los torchs
        ## y es el número de hidden nodes
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    ## Función de Contrastive Divergence para aproximar la likelihood del gradient
    ## (Optimizamos los weights para minimizar el outcome de la función de energía)
    def train(self, v0, vk, ph0, phk ):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk),0)
        self.a += torch.sum((ph0 - phk),0)
    
nv = len(training_set[0])      ## Las películas son los visible nodes
nh = 1682            ## El número de features (Actores, Oscares, etc..) por película que decidamos son los Hidden Nodes.
batch_size = 100
rbm = RBM(nv, nh)

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 ## Valor del loss
    c = 0.         ## Contador
    ## El entrenamiento se hace por un conjunto de usuarios (Batch_size) y NO unoXuno por eso se asigna un Step al for loop siguiente
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user : id_user+batch_size] ## Generamos nuestro Batch a partir de nuestro Training Set 
        v0 = training_set[id_user : id_user+batch_size] ## Creamos nuestro Target para poder conocer nuestro loss comparando predicciones con resultados esperados 
        ph0,_ = rbm.sample_h(v0) ## Obtenemos 
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            ## Evitar el update de los ratings sin calificación (Valor -1)
            ## Conservamos los valores -1 en el arreglo
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk) ## A partir de este punto, tenemos lo necesario para la función train
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) ## Loss Funtion es igual al mean de la diferencia de los valores absolutos existentes de v0 y vk
        c += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/c))
    
## Testing the RBM
test_loss = 0
c = 0.        
for id_user in range(nb_users):
    v = training_set[id_user : id_user+1]  
    vt = training_set[id_user : id_user+1]   
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) ## Loss Funtion es igual al mean de la diferencia de los valores absolutos existentes de v0 y vk
        c += 1.
print( 'test loss: '+str(test_loss/c))
   
    