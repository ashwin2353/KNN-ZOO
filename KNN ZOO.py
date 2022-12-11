# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:26:53 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("Zoo.csv")

df.dtypes
df.shape
df.head()

#===============================================================================
# data visualization
import seaborn as sns
sns.pairplot(df)
sns.heatmap(df.isnull(),cmap="Blues")

#==================
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(df["hair"],df["feathers"],color = "black")
plt.show()

#=======================

import numpy as np
Q1 = np.percentile(df["hair"],25)
Q1
Q2 = np.percentile(df["hair"],50)
Q3 = np.percentile(df["hair"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)

df[df["hair"]<LW]
df[df["hair"]>UW]

len(df[(df["hair"]<LW) | (df["hair"]>UW)])
# therefore 0 outlaiers

#====================================================================
# Split the variable as X and Y
X = df.iloc[:,1:17]
Y =df.iloc[:,17]

#====================================================================
# splitting the Data 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=23)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
#======================================================================
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
testing_accuracy = []

neighbors = range(1,31)

for number_of_neighbors in neighbors:
    KNN=KNeighborsClassifier(n_neighbors=number_of_neighbors)
    KNN.fit(X_train,Y_train)
    training_accuracy.append(KNN.score(X_train,Y_train))
    testing_accuracy.append(KNN.score(X_test,Y_test))

print(training_accuracy)    
print(testing_accuracy)

import matplotlib.pyplot as plt

plt.plot(neighbors,training_accuracy,label="training accuracy")
plt.plot(neighbors,testing_accuracy,label="testing accuracy")
plt.ylabel("Accuracy")
plt.slabel("number of neighbors")
plt.legend()
# therefore by seeing the plot i deside that K=19 the best value

#=======================================================================
# model fitting
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20,p=1)
knn.fit(X_train,Y_train)
Y_pred_train = knn.predict(X_train)
Y_pred_test = knn.predict(X_test)

#======================================================================
# metrics
from sklearn.metrics import confusion_matrix,accuracy_score
print("Training accuracy",accuracy_score(Y_train,Y_pred_train).round(3))
print("testing accuracy",accuracy_score(Y_test,Y_pred_test).round(3))

cm=confusion_matrix(Y_train,Y_pred_train)
cm=confusion_matrix(Y_test,Y_pred_test)
cm

#===================================================================
# validation set approch 
TrE = []
TsE = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    knn.fit(X_train,Y_train)
    Y_pred_train = knn.predict(X_train)
    Y_pred_test = knn.predict(X_test)
    TrE.append(accuracy_score(Y_train,Y_pred_train))
    TsE.append(accuracy_score(Y_test,Y_pred_test))

print(TrE)
print(TsE)
#===================================================================

# K-fold cross-validation
from sklearn.model_selection import KFold,cross_val_score
kfold = KFold(n_splits=5)
knn = KNeighborsClassifier(n_neighbors=5,p=1)
scores = cross_val_score(knn, X, Y, cv=kfold)

print("cross validation sores: ",scores)
print("average cv score: ",scores.mean())
print('Number of CV scores used in Average: ',len(scores))

#===============================================================














    

    




















