'''
Name - Gavish Gupta
Roll Number - 2018390
This .py file contains usage of scikit-learn implementation of logistic regression in Stochastic mode.
Necessary comments are made in order to make this code as readable as possible

Thank You!!
'''

#same code as used in test.py to split the dataset in 10 folds
def cross_validation_split(dataset, dataset_Y, folds=2):
	dataset_split = list()
	dataset_copy = list(dataset)
	dataset_copy_ans = list(dataset_Y)
	fold_size = int(len(dataset) / folds)
	dataset_ans = list()
	rng = random.seed(7)
	for i in range(folds):
		fold = list()
		fold_ans = list()
		while len(fold) < fold_size:

			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
			fold_ans.append(dataset_copy_ans.pop(index))
		dataset_split.append(fold)
		dataset_ans.append(fold_ans)
	return dataset_split, dataset_ans

#--------------------------------------------------------------------------------------------------------
#IMPORTING THE REQUIRED LIBRARIES 
from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import math
import random
import matplotlib.pyplot as plt
random.seed(7)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn import linear_model
from random import randrange
#---------------------------------------------------------------------------------------------------------

#importing the dataset as in scratch.py
req=[]
with open("/home/upriverbasil/Downloads/data_banknote_authentication.txt",'r') as reader:
	for line in reader.readlines():
		a = list(map(str, line.split(",")))
		l = []
		for i in a:
			l.append(float(i))
		req.append(l)
n = np.array(req,dtype=object)
X = n[:,:len(n[0])-1]
y = n[:,len(n[0])-1]

#Dividing the dataset into 7:1:2 samples
X, y = cross_validation_split(X,y,10)
X_o = X[0]
o = len(X)
for i in range(1,5):
	X_o = np.append(X_o,X[i],axis=0)
X_o = np.append(X_o,X[7],axis=0)
X_o = np.append(X_o,X[9],axis=0)
y_o = y[0]
for i in range(1,5):
	y_o = np.append(y_o,y[i])
y_o = np.append(y_o,y[7])
y_o = np.append(y_o,y[9])
Xtest = X[6]
Xtest = np.append(Xtest,X[8],axis=0)
ytest = y[6]
ytest = np.append(ytest,y[8])
Xval = []
Xval = np.array(X[5])
yval = np.array(y[5])
Xtrain=X_o
ytrain=y_o

#using scikit learn SGD classifier with the used alpha and iter_values values
model = linear_model.SGDClassifier(max_iter = 1000,alpha = 10, random_state = 7)
model.fit(Xtrain,ytrain)
print(model.score(Xtest,ytest), model.score(Xtrain,ytrain))

