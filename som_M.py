## Self Organizing Map
## Detecting Fraud

import numpy as np
import matplotlib as plt
import pandas as pd

## Importing the dataset

dataset = pd.read_csv("Credit_Card_Applications.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1]
## Esto es solo para separar la data porque en Unsupervised Deep Learning no necesitamos en realidad
## una dependant variable

## Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

## Training the SOM
from minisom import MiniSom
som = MiniSom( x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5,)
## Params = 0) X y Y = Tammaño del SOM
##          1) input_len = Número de Inputs 
##          2) sigma = Radio de los "Clusters" en el grid
##          3) learning_rate = Decide que tanto se ajustan las cooredenadas (weights))

## Initializing the weights and Trainig the SOM
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

##Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
## som.distance_map regresa un vector con los valores de los Mean Interneural Distances
colorbar()
markers = ["o","s"]
colors = ["r", "g"]
for i, c in enumerate(x):
    w = som.winner(c)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], 
         markeredgecolor = colors [y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
    ## Markers y[i] toma el valor uno o cero según se apropbo o no al customer
    ## para saber si cera un cuadrado o un círculo

## Findind the Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
## Se utilizan las coordenadas (de mappings) de posibles fraudes en el mapa
frauds = sc.inverse_transform(frauds)