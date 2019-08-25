import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler,LabelEncoder
import time

import csv

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten, LSTM, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical, np_utils
from keras import regularizers


data_path = "/Users/katherinele/CISC452/Project/Data/"
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test/test.csv")

#print(train_df.describe())

#print(train_df.dtypes)

target = train_df['AdoptionSpeed']
features = train_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
features_test = test_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])

# standardize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
features[ :1]
print(features)

# split features and targets into train/test sets
# random_state set at 14 or 21 for specific results

X_train, X_test, Y_train, Y_test = train_test_split(
        features, target, random_state=21)

# encode class values as integers
Y_test_1h = to_categorical(Y_test)

Y_train_1h = to_categorical(Y_train)

#first input layer should be the number of input features
#output layer should be number of classes
number_of_features = 19
model = Sequential()
model.add(Dense(19, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(250, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(75, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(50, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(5, activation='softmax'))

#Compile neural network
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy']
             )

#Train Neural net
#stime = time.monotonic()
history = model.fit(X_train, Y_train_1h, epochs=500, batch_size=19)
#etime = time.monotonic()

#print('training time (s): ', etime-stime)

score, acc = model.evaluate(X_test, Y_test_1h,
                            verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)
# standardize test features
scaler = MinMaxScaler()
test_features = scaler.fit_transform(features_test)
predicted_target = model.predict(test_features)
y_pred = pd.DataFrame(model.predict_classes(test_features), index = test_df.index)
y_pred['PetID'] = test_df['PetID']
y_pred.rename(columns={0:'AdoptionSpeed'}, inplace=True)
y_pred = y_pred[['PetID','AdoptionSpeed']]


y_pred.to_csv('results.csv', index=False)
