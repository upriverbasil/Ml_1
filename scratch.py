'''
Name - Gavish Gupta
Roll Number - 2018390
This .py file contains self implementation of linear regression using RSME and MAE losses and logistic
regression in Batch mode and Stochastic mode.
Necessary comments are made in order to make this code as readable as possible

Thank You!!


NOTE: I have not used vectorized form of gradient descent I have used building the equation form and then
summing over all the values.
'''


#----------------------------------------------------------------------------------------------
#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import random
#-----------------------------------------------------------------------------------------------

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
    	pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X= []
        y = []
        
        y = y
        if dataset == 0:
            # Implement for the abalone dataset
            req = []
            #reading the file mentioned in the path
            with open("/home/upriverbasil/Downloads/abalone/Dataset.data",'r') as reader:
            	for line in reader.readlines():
            		a = list(map(str, line.split(" ")))
            		l = []
            		for i in a:
            			#using one hot encoding that is 100 for M, 010 for F, 001 for I
            			if(i=="M"):
            				l.append(1)
            				l.append(0)
            				l.append(0)
            			elif(i=="F"):
            				l.append(0)
            				l.append(1)
            				l.append(0)
            			elif(i=="I"):
            				l.append(0)
            				l.append(0)
            				l.append(1)
            			else:
            				l.append(float(i))
            		req.append(l)
            #converting the list to numpy array
            n = np.array(req,dtype=object)
            X = n[:,:len(n[0])-1]
            y = n[:,len(n[0])-1]
        elif dataset == 1:
        	req = []
        	with open("/home/upriverbasil/Downloads/VideoGameDataset.csv",'r') as reader:
        		count1 = 0 #variable used to skip the first column which contains the column names
        		for line in reader.readlines():
        			if(count1==0):
        				count1+=1
        				continue
        			a = list(map(str, line.split(",")))
        			l = []
        			count = 0
        			for i in a:
        				#count indicates the column number for each list 9th indicates Global_Sales, Critic_Score, User_Score 
        				if(count==9 or count==10 or count==12):
        					#try catch used for string or NaN values
        					try:
        						l.append(float(i))
        						count+=1 
        					except:  
        						l.append(i)
        						count+=1
        						continue
        				else:
        					count+=1
        			req.append(l)
        	n = np.array(req,dtype=object)
        	#variables to calculate means of the three columns 1-> Global_Score, 2-> Critic_Score, 3-> User_Score
        	count1 = 0
        	count2 = 0
        	count3 = 0
        	mean1 = 0
        	mean2 = 0
        	mean3 = 0
        	for j in range(len(n[0])):
        		for i in range(len(n)):
        			if(isinstance(n[i,j],str) or math.isnan(float(n[i,j]))):
        				continue
        			else:
        				if(j==0):
        					count1+=1
        					mean1+=n[i,j]
        				elif(j==1):
        					count2+=1
        					mean2+=n[i,j]
        				elif(j==2):
        					count3+=1
        					mean3+=n[i,j]
        	mean2/=count2
        	mean3/=count3
        	#loop to remove the values with the mean column in the dataset
        	for j in range(len(n[0])):
        		for i in range(len(n)):
        			if(isinstance(n[i,j],str) or math.isnan(float(n[i,j]))):
        				if(j==1):
        					n[i,j] = mean2
        				else:
        					n[i,j] = mean3
        				continue
        	print(mean2,mean3,n)
        	X = n[:,1:3]
        	y = n[:,0]
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            req = []
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
            print(X,y)
            pass

        return X, y


#-----------------------------------------------------------------------------------------------------

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
    	self.theta = [] #theta indicates the n+1 weights of the function required to build a linear regression model
    	self.training_loss = [] #training loss used to plot the Training_loss Vs iterations graph
    	self.validation_loss = [] #validation loss used to plot the Validation_loss Vs iterations graph

    #here loss function means the hypothesis which is used in calculating the loss function
    def loss_func(self,x,m):
        h = 0
        for i in range(len(x)+1):
            if(i==0):
                h += self.theta[i]*1
            else:
                h += self.theta[i]*x[i-1]
        return h

    #here val_loss gives the validation loss on X for the given loss function
    def val_loss(self,X,y,theta,loss="RMSE"):
    	error = 0
    	if(loss=="RMSE"):
    		# print("RMSE")
    		for j in range(len(X)):
    			h = theta[0]
    			for i in range(1,len(X[j])+1):
    				h+=theta[i]*X[j,i-1]
    			error=error+math.pow(h-y[j],2)
    	else:
    		for j in range(len(X)):
    			h = theta[0]
    			for i in range(1,len(X[j])+1):
    				h+=theta[i]*X[j,i-1]
    			error=error+abs(h-y[j])
    	return error

    #function used to plot the required graphs
    def plot_a(self):
    	plt.plot(self.training_loss,label='training loss')
    	plt.xlabel("No. of iterations")
    	plt.legend()
    	plt.show()

    	plt.plot(self.validation_loss, label="validation loss",color="red")
    	plt.xlabel("No. of iterations")
    	plt.legend()
    	plt.show()
    	# print(self.validation_loss)

    #function used to calculate the train and test loss using normal equation
    def normalform(self, X, y, x_test, y_test):
    	#X is of the form [[X1],[X2],..,[XK]..,[XN]] where Xi represents one of the groups from the K-folds
    	#Y is of the similar form
    	#So the following code is to convert it to usable form
    	if(X==[]):
    		return None
    	o = len(X)
    	X_o = X[0]
    	for i in range(1,o):
    		X_o = np.append(X_o,X[i],axis=0)
    	X = X_o
    	y_o = y[0]
    	for i in range(1,o):
    		y_o = np.append(y_o,y[i])
    	y = y_o
    	y = y.transpose()
    	# print(X.shape,y.shape)
    	Xabcd = copy.deepcopy(X)
    	#Xabcd represents the training matrix so now adding a new column with new ones to make it n+1 dim as theta is of the size n+1 dim
    	Xabcd = np.c_[np.ones((len(X),1)),X]
    	#Using fomula Theta = inverse(X'.X).(X'.Y) here X' represents the Transpose of X
    	Xo = Xabcd.transpose().dot(Xabcd)
    	Xo = Xo.astype(np.float64)
    	xinv = np.linalg.inv(Xo)
    	theta = xinv.dot(Xabcd.transpose()).dot(y)
    	#To calculate the predicted values of X on basis of new theta 
    	Xabcd = np.c_[np.ones((len(X),1)),X]
    	ypred = Xabcd.dot(theta)
    	error = 0
    	# print(y.shape,ypred.shape,"lll")
    	#calculating train_loss
    	for i in range(len(ypred)):
    		error+=(ypred[i]-y[i])**2
    	train_loss = error/len(X)
    	train_loss = error**0.5
    	# print(ypred)
    	#To calculate the predicted values of X_test on basis of new theta 
    	Xabcd = np.c_[np.ones((len(x_test),1)),x_test]
    	ypred = Xabcd.dot(theta)
    	error = 0
    	# print(y.shape,ypred.shape,"lll")
    	#calculating test_loss
    	for i in range(len(ypred)):
    		error+=(ypred[i]-y_test[i])**2
    	test_loss = error/len(x_test)
    	test_loss = error**0.5
    	return train_loss,test_loss

    def fit(self, X, y, alpha = 0.00001	, no_of_iterations = 10, loss = "RMSE",x_test=[],y_test=[]):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        #similar processing of matrix as done in the normalform()

       	if(X==[]):
            return None
        o = len(X)
        X_o = X[0]
        for i in range(1,o):
        	X_o = np.append(X_o,X[i],axis=0)
        X = X_o
        n = len(X[0])
        for i in range(n+1):
            self.theta.append(0)
        o = len(y)
        y_o = y[0]
        for i in range(1,o):
        	y_o = np.append(y_o,y[i])
        y = y_o
        y = y.transpose()


        for lmn in range(no_of_iterations):
            h = 0
            der_loss = [] #der_loss to store gradient in gradient descent for each n+1 weights
            for i in range(n+1):
                der_loss.append(0)
            under_root = 0 #under_root is the training loss


            #for calculating the derivative of RMSE loss and else part for derivative for MAE loss
            #for RMSE loss the derivative of cost function is diff than MAE so two separate functions have to be used
            if(loss=="RMSE"):
                for i in range(len(X)):
                    h = self.loss_func( X[i],len(X))-y[i]
                    under_root+=math.pow(h,2)
                    #to calculate batch gradient descent over all the n samples
                    for j in range(n+1):
                        if(j==0):
                           
                            der_loss[j]+=h
                        else:
                            der_loss[j]+=h*X[i,j-1]
                        

            else:
                 for i in range(len(X)):
                    h = self.loss_func( X[i],len(X))-y[i]
                    under_root+=abs(h) 
                    #to calculate batch gradient descent over all the n samples
                    for j in range(n+1):
                        if(j==0):
                            temp1 = 1
                            if(h<abs(h)):
                            	temp1=-1
                            der_loss[j]+=temp1
                        else:
                        	
                        	temp1 = 1
                        	if(h<abs(h)):
                        		temp1=-1
                        	der_loss[j]+=temp1*X[i,j-1]



            #this is else block is made to add values in the self.training_loss and self.validation_loss used in plotting of graphs which are required 
            if(loss=="RMSE" and len(x_test)!=0):
            	under_root = under_root/len(X)
            	under_root = math.pow(under_root,1/2)
            	self.training_loss.append(under_root)

            	under_root_1 = self.val_loss(x_test,y_test,self.theta) #to calculate the validation loss for plotting graphs
            	under_root_1 = under_root_1/len(x_test)
            	under_root_1 = math.pow(under_root_1,1/2)
            	self.validation_loss.append(under_root_1)
            	
            elif(len(x_test)!=0):

            	under_root = under_root/len(X)
            	self.training_loss.append(under_root)

            	under_root_1 = self.val_loss(x_test,y_test,self.theta,loss="MAE")
            	under_root_1 = under_root_1/len(x_test)
            	self.validation_loss.append(under_root_1)


            #to change the value of theta according to the gradient
            for j in range(n+1):
            	temp = alpha*der_loss[j]/len(X)
            	if(loss=="RMSE"):
	            	temp/=under_root
            	if(j==0):
            		self.theta[j] = (self.theta[j] - temp)
            	else:
            		self.theta[j] = (self.theta[j] - temp)

        return self



    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        ans : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        # return the numpy array ans which contains the predicted values
        
        ans = []
        for j in range(len(X)):
            h = self.theta[0]
            #calculating ax0 + bx1 + cx2 +..... where a, b, c are the weights
            for i in range(1,len(X[j])+1):
                h+=self.theta[i]*X[j,i-1]
            ans.append(h)
        return ans

#--------------------------------------------------------------------------------------------------------


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """
    #variables and functions in this class have use as in the class LinearRegression wherever there's a change that will be mentioned
    def __init__(self):
    	self.theta = []
    	self.training_loss = []
    	self.validation_loss = []

    def loss_func(self,x,m):
        h = 0
        for i in range(len(x)+1):
            if(i==0):
                h += self.theta[i]*1
            else:
                h += self.theta[i]*x[i-1]
        return h

    def val_loss(self,X,y):
    	#so validation loss is calculated here it is modified in comparison to that in LinearRegression because of the change in hypothesis
    	error = 0 #to over all the errors 
    	for j in range(len(X)):

    		h = self.theta[0]

    		for i in range(1,len(X[j])+1):
    			h+=self.theta[i]*X[j,i-1]

    		h = 1/(1+math.pow(2.71,-1*h))

    		#this if else block is used to handle the math domain error due to log(0)
    		if(h==0):
    			h=0.00000000000000001
    		elif(h==1):
    			h=0.99999999999999999

    		lo = -1*((y[j])*(math.log(h))+(1-y[j])*(math.log(1-h))) #loss function in logistic regression
    		error+=lo
    	return error

    #not preferable to use to plot training loss in SGD mode
    def plot_a(self):
    	plt.plot(self.training_loss,label='training loss')
    	plt.show()
    	plt.xlabel("No. of iterations")
    	plt.plot(self.validation_loss, label="validation loss",color="red")
    	plt.legend()
    	plt.show()


    def fit(self, X, y, alpha =30, no_of_iterations = 1000, func = "SGD",x_test=[],y_test=[]):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py

        n = len(X[0])
        for i in range(n+1):
            self.theta.append(0)
        y = y.transpose()
        # print(X,n, X.shape)
        for lmn in range(no_of_iterations):
            h = 0
            der_loss = []
            for i in range(n+1):
                der_loss.append(0)
            under_root = 0 #to calculate loss as in LinearRegression
            #This is else block is used to determine SGD or BGD by determining the number of samples the loss will be calculated upon at a time
            
            if(func == "SGD"):
            	r1 = random.randint(0,len(X)-1)
            	r2 = r1+1
            else:
            	r1 = 0
            	r2 = len(X)

            for i in range(r1,r2):
            	ho = 1/(1+math.pow(2.7182,-1*(self.loss_func( X[i],len(X)))))
            	h = ho
            	#to handle log(0)
            	if(ho==0):
            		ho=0.000000000000000000001
            	if(ho==1):
            		ho=0.999999999999999999999
            	# print(ho)
            	under_root+=-1*( ((y[i])*math.log(ho)) + ((1-y[i])*math.log(1-ho)) ) #loss function
            	h = ho-y[i]
            	
            	for j in range(n+1):
            		# try:
            		if(j==0):
            			der_loss[j]+=h
            		else:
            			der_loss[j]+=h*X[i,j-1]

           	#for plotting
            under_root = under_root/len(X)
            self.training_loss.append(under_root)
            if(len(x_test)!=0):
	            self.validation_loss.append(self.val_loss(x_test,y_test)/len(x_test));

            #for revising the weightss
            for j in range(n+1):
            	temp = alpha*der_loss[j]/len(X)
            	if(j==0):
            		self.theta[j] = (self.theta[j] - temp)
            	else:
            		self.theta[j] = (self.theta[j] - temp)

        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        ans : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array ans which contains the predicted values
        # X = np.insert(X,0,np.ones(len(X)),axis=1)
        # return X.dot(self.theta.transpose())

        ans = []
        for j in range(len(X)):
            h = self.theta[0]
            for i in range(1,len(X[j])+1):
                h+=self.theta[i]*X[j,i-1]
            val = 1+math.pow(2.71,-1*h) #to calculate hypothesis H(x) = 1/(1+e^(-theta.X))
            ans.append((1/val))

        return ans




#-------------------------------------END OF CODE----------------------------------------------------------