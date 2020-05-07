#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create dataset and separate independent and dependent features

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13]
Y=dataset.iloc[:,13]

#create dummy variables

geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#concatenate X,gender,geography

X=pd.concat([X,gender,geography],axis=1)

#drop Gender and Geography from X since you do not need them 

X=X.drop(['Geography','Gender'],axis=1)

#split the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#do feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#import keras library

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU,PReLU,ELU

#initialise the ANN

classifier=Sequential()

#merging the input layer and the first hidden layer

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

#merging the second hidden layer

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

#merging the output layer

classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))

#compile the ANN

classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

#fitting the ANN to the training set

model_history=classifier.fit(X_train,Y_train,batch_size=10,validation_split=0.33,nb_epoch=100)

#predict the test set results

Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)

#make the confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#calculate accuracy

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,Y_pred)






