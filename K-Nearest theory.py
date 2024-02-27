# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 08:13:21 2024

@author: HP
"""

'''
#######k-nearest n are shallow  model meaning no hypothesis or 
model will be prepaired new base classifire probablistic/ statstical mode
#######KNN is also called instance based model meaning stored training instatnces themselves 
represent the knowldge'''

'''
similarity and Dissimilarity
Similarity
-nemerical measures of how alike two data object are
-is higher when objects are more alike

Dissimilarity
-numerical measures of how diffrent are two data objects
-lower when objects are more alike
-minimum dissimilarity is often 0
-upper limit varies

Proximity referes to a similarity or dissimilarity
 Ecludian distance
 city block distance
 Minkowski Distance:- it is generalization of euclidian distance
 
'''

'''
How shall i choose the value of k in KNN Algorithm
-K in KNN algorithm is based on feature similarity chossing the right
value of k is a process called paramiter tuniing and is imp for better accuracy

-Finding the value of k is note easy

-there is no stuctured method to find the best value for ''K''
we need to find out with various values by trial and error assuming that 
training data is unknown.

-1. Find K-nearest pont to Xq in D
-2. Fi.nd out the curresponding value of X1, X2, X3 and its label Y1, Y2, Y3
-3. Then apply majority vote


'''
####################################################################3#

'''
lets assume there are 10 images 
dog1  dog2  dog3   dog4    dog5  mobile   home  dog6    dog7 dog8
dog   dog   dog    no dog   dog  no dog   dog   no dog  dog  dog
'''
'''
positive class prediction:- done by ML algorithm as dog
Negative class prediction:- done by ML algorithem as no dog


in positive class prediction 4 results are correct predicted (TP)
in false class prediction it has not dog but it has predicted dog
3 results(FN)

TN:-the images no dog prediction is negstive class prediction

out of 10 prediction right result is 5
Accuracy:- 5/10:- 0.5

When calculating prediction how many you got it right?

precision=4/7=0.57
precision= TP/(TP+FP)



'''

"""
Whenever you are going to calculate Recall is out of all dog truth how many
you got it right
total dog truth samples=6
Recall=4/6=0.67
True positiove=4
Recall=TP/(TP+FN)


"""

"""
For precision think about predictions as your base

For recall think about truth as your base
"""

"""
F1 Score
It is evaluation matrix that measurres a models accuracy. It combines the
the precision and recall score of a model
"""

import pandas as pd
import numpy as np
wbcd=pd.read_csv("wbcd.csv")

#there are 569 rows and 32 columns
wbcd.describe()
#in output column there is only 8 for Benien and M for
#maligananat
#let us first convert is as benign and  malignant
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='8','Beniegn',wbcd['diagnosis'])
#in wbcd there is a column named 'diagnosis' where ever there is '8' replace value with Benign
#similarity where ever there is M in the same columns replace with 'Malignant'
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Malignant',wbcd['diagnosis'])
###################################################################
#0th column is patient ID us drop it
wbcd=wbcd.iloc[:,1:32]
##################################################################
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now let us apply this function to the dataframe

wbcd_n=norm_func(wbcd.iloc[:,1:32])
#Because now 0th column is output or label it is not considered hence
#1:all
####################################################################
#let us now apply X as input and Y as a output
X=np.array(wbcd_n.iloc[:,:])
##slice in wbcd_n, we are allready execluding output column, hence all rows
#and 
Y=np.array(wbcd['diagnosis'])

####################################
#Noe let us split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test=train_test_split(X,Y,test_size=0.2)

#here you are passing X,Y instead dataframe handle
#there could chances of unbalancing  of data
#let us assume u have 100 data points, out of which 80Nc and 20 cancer
#there data points must be equaly distributed
#there is tstified sampling concept is used
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred=knn.predict(X_test)
pred
##########################################################
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
pd.crosstab(pred,Y_test)

#let us check the applicability of model
#i.e miss classification,actual patient is malignant
#i.e cancer paitent but predicted is Benien is 1
#actual patient is Benien and predicted as cancer patient is 5
#hence this model is not acceptable
#########################################################
#let us try to select correct value of K
acc=[]
#Running knn algorithm for k=3 to 50 in the step of 2
#k value selected is odd value
for i in range(3,50,2):
    #declare the model
    neigh=KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train,Y_train)
    train_acc=np.mean(neigh.predict(X_train)==Y_train)
    test_acc=np.mean(neigh.predict(X_test)==Y_test)
    acc.append([train_acc,test_acc])
#if you will see the acc, it has got two accuracy, i[0]-train_acc
#i[1]=test_acc
#to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
#There are 3,5,7, and 9 are possible values where accuracy is good
#Let us check for k=3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
pred=knn.predict(X_test)
accuracy_score(pred,Y_test)
pd.crosstab(pred,Y_test)
#i.e miss classification actual patient is malignant
#i.e cancer patient but predicted is Benien is 1
#Actual patient is Benien and predicted as cancer patient is 2
#hence this model is not acceptable
#for 5 same sinario
#for k=7 we are geetting zero false
###########################################################






