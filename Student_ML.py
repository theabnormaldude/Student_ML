# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 02:45:17 2019

@author: Shiv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:/Code/student/student-mat.csv", sep=";")
data=data[["G1","G2","G3","studytime","failures","absences"]]

predict="G3"

X=np.array(data.drop([predict],1))
Y=np.array(data[predict])

x_train, x_test,y_train, y_test=train_test_split(X,Y, test_size=0.1)

linear=LinearRegression()
linear.fit(x_train,y_train)
acc=linear.score(x_test, y_test)
pred=linear.predict(x_test)
print(acc)
for x in range(len(pred)):
    print(pred[x],x_test[x], y_test[x])
print("Co: ",linear.coef_)  #Higher the coefficient, higher the weight 
print("Intercept: ", linear.intercept_)
