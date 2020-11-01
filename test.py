'''
Name - Gavish Gupta
Roll Number - 2018390
This .py file contains usage of self implemented methods in scratch.py and also it contains self implemented
code of k folds cross validation
Necessary comments are made in order to make this code as readable as possible

Thank You!!
'''

#-------------------------------------------------------------------------------------------------
#Function for cross validation splits

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
	return dataset_split,dataset_ans

#--------------------------------------------------------------------------------------------------

#IMPORTING THE REQUIRED LIBRARIES 
from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from random import randrange
random.seed(7) #SETTING THE SEED TO PRODUCE REPRODUCIBLE RESULTS

#--------------------------------------------------------------------------------------------------

preprocessor = MyPreProcessor()
print('Linear Regression')
X, y = preprocessor.pre_process(1)

best_error = 1000000000 #best error represent the best error if the code is run for multiple folds
best_fold = 0           # fold which gives the best(least) error
preprocessor = MyPreProcessor()
print('Logistic Regression')
X, y = preprocessor.pre_process(1)
for fold_no in range(5,9):
	best_error = 10e100
	error = 0
	best_fold = 0

	#splitting of data into fold_no folds
	X, y = cross_validation_split(X,y,fold_no)

	for i in range(len(X)):
		#using ith fold out of fold_no as testing one
		Xtrain = np.array(X[:i]+X[i+1:])
		ytrain = np.array(y[:i]+y[i+1:])
		Xtest = np.array(X[i])
		ytest = np.array(y[i])
		linear = MyLinearRegression()
		loss = "RMSE" #change to MAE if required

		#linear.fit can be used without x_test, y_test and loss
		# linear.fit(Xtrain,ytrain,x_test = Xtest, y_test = ytest,loss = loss)
		linear.fit(Xtrain,ytrain)

		# print(linear.normalform(Xtrain,ytrain,Xtest,ytest)) #to get normal form results. uncomment this line to use
		
		ypred=[]
		ypred = linear.predict(Xtest) #to predict values for Xtest
		print("Predicted values are ",ypred)
		for i in range(len(ypred)):
			if(loss=="RMSE"):
				error+=abs(ytest[i]-ypred[i])**2
			else:
				error+=abs(ytest[i]-ypred[i])
		
	# linear.plot_a() #use this to plot only when in linear.fit() xtest and ytest are also given
	error/=fold_no
	error/=len(ypred)
	if(loss=="RMSE"):
		error=error**(0.5)
	#print("error ",error,"fold no ",fold_no) #uncomment this line to print error
# 	if(error<best_error):
# 		best_error = error
# 		best_fold = abcd
# print(best_fold,best_error) #uncomment this line to print best error in all the folds

print('Logistic Regression')
best_error = 1000000000
best_fold = 0
preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(2)
print(X.shape,y.shape)
X, y = cross_validation_split(X,y,10)
# Create your k-fold splits or train-val-test splits as required
X_o = X[0]
o = len(X)
print(o)
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

logistic = MyLogisticRegression()
#logistic.fit() can be used without x_test and y_test
# logistic.fit(Xtrain,ytrain,x_test = Xtrain, y_test = ytrain, func = "SGD")
logistic.fit(Xtrain,ytrain)
# logistic.plot_a() #dont use this for plotting training losses with SGD

ypred = logistic.predict(Xtest)
print("Predicted values are ",ypred)
arg = []
error = 0
count = 0
for i in range(len(ypred)):
	ho = ypred[i]
	if(ho>=0.5):
		if(ytest[i]==1):
			count+=1
	else:
		if(ytest[i]==0):
			count+=1
		else:
			count = count
			
count/=len(ytest)
print(count)#testing loss
ypred = logistic.predict(Xtrain)
arg = []
count = 0
for i in range(len(ypred)):
	ho = ypred[i]
	if(ho>=0.5):
		if(ytrain[i]==1):
			# print(ytest)
			count+=1
	else:
		if(ytrain[i]==0):
			count+=1
count/=len(ytrain)
print(count)#training loss

