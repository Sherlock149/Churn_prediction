# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:32:58 2021

@author: abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ann2\dataset\Churn_Modelling.csv')
#there are 13 independent features and 1 dependent feature but row no. and customer ID doesn't
#play any role

#storing the fetures X
X = dataset.iloc[:,3:13]

#the dependent feature y
Y = dataset.iloc[:,13]

#creating dummy variable for geography and gender as they are non numeric

geography = pd.get_dummies(X['Geography'], drop_first=(True));
gender = pd.get_dummies(X['Gender'],drop_first=(True));

#concat the above and remove the columns containing strings

X=pd.concat([X,geography,gender],axis=1)
X = X.drop(['Geography','Gender'],axis=1)
# now we have 11 independent features in X
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
#fit_trnasform first fit() then transform(). During fit() we get mean & deviation
#next time only transform is used cuz it uses the same fit() parameters
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating the ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential #must
from tensorflow.keras.layers import Dense #for adding the hidden layers
from tensorflow.keras.layers import Dropout #for enabling dropout

#initialiser
classifier = Sequential()

#units = output parameters ie no. of neurons in next layer use 
#hyperparameter tuning to determine this value
classifier.add(Dense(units=14, kernel_initializer='he_uniform',input_dim=11,activation='relu'))
classifier.add(Dense(units=32, kernel_initializer='he_uniform',activation='relu'))
classifier.add(Dense(units=36, kernel_initializer='he_uniform',activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))


#to add dropout with ratio p
#classifier.add(Dropout(p))
#for very big data

classifier.add(Dense(units=1, kernel_initializer='glorot_uniform',activation='sigmoid'))
# the multilayers are ready
# compiling the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=('accuracy'))

from tensorflow.keras.models import load_model
#fitting the ann acc to the training dataset
#validation is basically trying to test the model with a fraction of X_train
model_history = classifier.fit(X_train,Y_train,batch_size=10,epochs=10,validation_split=0.33)
#we store it in a history for plotting purpose later

classifier.save('ann2/ann2_model.h5')

#using the classifier with the test data
Y_predict = classifier.predict(X_test)
ptr = 0
for i in Y_predict:
    if i > 0.5:
        Y_predict[ptr] = 1
    else:
        Y_predict[ptr] = 0
    ptr += 1

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(Y_test, Y_predict)
score = accuracy_score(Y_test, Y_predict)

#plotting
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(['train','validation'], loc='upper left')
plt.title('Model Accuracy')
plt.show()
print("Test Accuracy:",score*100,"%")
