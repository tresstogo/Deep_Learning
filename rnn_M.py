## Recurrent Neural Network

## Part 1: Data Preprocessing
##Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##Importing the Training Set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) 
## Rango del 1 al 0 para el normalizado 
training_set_scaled = sc.fit_transform(training_set)
##Number of Time Steps = Hace referencia a lo que nuestra RNN será capaz de recordar
## Number of Time Steps = Cantidad de eventos pasados de recordar para tomar en cuenta para realizar una predicción.

## Creating a data structure with 60 timesteps (x values) and 1 output (y value)
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

## Reshaping (adding a new dimension) Formato 3D
x_train.shape[0], x_train.shape[1],
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1 ))
## Segundo argumento para el reshape son: 1) Batch_Size = Número de Rows
                                     ##   2) Time_Steps = Número de Columns
                                     ##   3) input_dim = Número de indicadores extra (Nueva capa 3D)
                                     
## Part 2: Building the RNN
## Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

## Initializing te RNN
regressor = Sequential()  

## Adding te first LSTM layer and Regularisation DropOut
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1)))                                  
## LSTM Parameters: 1) units = Número de Neuronas
                ##  2) return_sequences = Loop is activated
                ##  3) input_shape = Dimensions (Time_Steps y Número de Indicadores)
## Dropping Neurons Randomly (20% of them) with Dropout
regressor.add(Dropout(0.2))  

##Second LSTM Layer
regressor.add(LSTM(units = 50, return_sequences=True))                           
regressor.add(Dropout(0.2))

##Third LSTM Layer
regressor.add(LSTM(units = 50, return_sequences=True))                           
regressor.add(Dropout(0.2))
             
##Fourth LSTM Layer
regressor.add(LSTM(units = 50))                           
regressor.add(Dropout(0.2))

##Output Layer (Using Dense for a fully connected layer)
regressor.add(Dense(units = 1)) 

## Compiling RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")                     

## Fitting the RNN to the Training_Set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

## Part 3: Making the predictions and visualizing results

## Getting the real stock price
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values
## No podemos hacer scaling al real_stock_price ya que los datos de test nunca deben ser modificados
## por lo que se realizará la concatenación directo de los datasets (training y test)

## Getting the predicted stock price of 2017

## Para realizar la evaluación completa en el test_set de nuestro regressor
## es necesario contar con la data previa al dataset_test, por lo que es requerido concatenar
## ambos datasets para poder hacer una predicción adecuado por lo que procedemos a utilizar pandas para la función concat
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0) ##axis indica: concatenar filas o columnas
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
## Se le hace este indexing a dataset_total para sacar los inputs necesarios de los últimos 3 meses 
## Debido a que por mes solo se toman en cuenta los días entre semana para las acciones de Google por lo que son 20 días por mes
## A partir de que se consigue el lower bound con el menos 60 todo en adelante son inputs a considerar para la predicción

##Se le da el formato correcto con reshape y se escala con el objeto de la clase Scale, previamente utilizado
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
## Se reestructuran los inputs en una lista
x_test = []
for i in range(60,80):
    x_test.append(inputs[1-60:i, 0])
x_test = np.array(x_test)
## Finalmente le damos el el formato 3D como fue realizado previamente
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

##Código de Predicción: (Se regresa a la escala inicial)
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


## Visualizing theresults
plt.plot(real_stock_price, color = "red", label = "Google Real Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Google Predicted Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()