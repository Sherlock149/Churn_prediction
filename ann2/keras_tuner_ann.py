# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:33:25 2021

@author: abhishek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


# Binary Classification dataset
dataset = pd.read_csv('ann2\dataset\Churn_Modelling.csv')
# X = independent features
# Y = dependent features
X = dataset.iloc[:,3:13]
Y = dataset.iloc[:,13]

#creating dummy variable for geography and gender as they are non numeric
geography = pd.get_dummies(X['Geography'], drop_first=(True));
gender = pd.get_dummies(X['Gender'],drop_first=(True));

#concat the above and remove the columns containing strings
X=pd.concat([X,geography,gender],axis=1)
X = X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
The below function is copied from keras documentation to do the hyperparameter
tuning. Change the activation function, loss function and metric acc to the
problem statement. Few points:
    num_layer: Determines the number of hidden layers. default: 2 - 20
    units: Determines the number of neurons. default: 32 - 512 (step size: 32)
    hp.Choice: Takes discrete values in that order
    hp.Int: Tests all integers in that range
"""

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=6,
                                            max_value=40,
                                            step=2),
                               activation='relu'))
        
    # last layer: Chage acc to dataset
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

# Running randomSearch acc to the above model and saving the results in tuner
# Copied from keras documentation. 
# Available tuners: RandomSearch and Hyperband

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='ann2',
    project_name='tuner_summary')


tuner.search_space_summary()

tuner.search(X_train, Y_train, epochs=10, validation_data = (X_test,Y_test))
tuner.results_summary()
