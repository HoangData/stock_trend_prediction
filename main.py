import os

import keras
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler # used for feature scaling
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'





df = pd.read_csv('VNINDEX.csv')



data_train = df[df['DATE']<'2019-01-02'].copy()
data_test = df[df['DATE']>='2019-01-02'].copy()
data_training=data_train.copy()

past_60_days = data_training.tail(60)
data_test = past_60_days.append(data_test, ignore_index = True)

# Dropping 'DATE'
data_train = data_train.drop(['DATE'], axis = 1)
data_test = data_test.drop(['DATE'], axis = 1)

sc = MinMaxScaler()
data_train= sc.fit_transform(data_train)
data_test = sc.transform(data_test)

X_train = []
y_train = []

for i in range(60, data_train.shape[0]):
    X_train.append(data_train[i - 60:i])
    y_train.append(data_train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []

for i in range(60, data_test.shape[0]):
    X_test.append(data_test[i-60:i])
    y_test.append(data_test[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

'''
model = Sequential()

# Adding the first RNN layer and some Dropout regularisation
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))
# Adding a second RNN layer and some Dropout regularisation.
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
# Adding a third RNN layer and some Dropout regularisation.
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth RNN layer and some Dropout regularisation.
model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 50, batch_size = 32)

model.save("mymodel")'''

model = keras.models.load_model("mymodel")

y_pred = model.predict(X_test)
print(y_pred)

