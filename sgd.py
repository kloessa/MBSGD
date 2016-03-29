# sgd.py

import numpy as np
import time
from numpy import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys


def hinge(theta, X, y, lda):

	"""
	Hinge loss implementation.
		
	input:	1. Parameter array - theta
		2. Matrix with features - X
		3. Array with targets - y
		4. Regularization parameter lambda - lda			

	output:	1. Hinge loss - cost 
		2. Hinge gradient - grad
	"""

# Initialize parameters
        m, n = X.shape					
	cost, v = 0, 0
	grad = np.zeros(n)

# Calculate Hinge loss with l2 regularization      
	for i in range(0, m):				
    		v = y[i] * np.dot(theta, X[i])
		cost += (1/2) * max(0, 1 - v) ** 2 + (lda * sum(theta[1:] ** 2))# Not regularize intercept

# Calculate gradient for Hinge
    		if v > 1:				
			grad += 0
		else:
			grad -= y[i] * X[i] - 2 * lda * theta
			grad[0] -= y[i] * X[i, 0]			# Not regularize intercept
    	return (cost, grad)


def pred_hinge(X, theta):

	"""
	Prediction function for Hinge loss.

	input:	1. Feature array - theta
		2. Matrix with examples - X
				
	output:	Array with predicted values - pred
	"""

	m, n = X.shape
    	pred = np.zeros(m)
	pred = np.dot(X,theta)
	pred[pred < 0] = -1
	pred[pred > 0] = 1
	return (pred)	


def mbsgd(X, y, X_test, y_test, steps, thresh, lda, batch, eta):

	"""
	Implementation of mini-batch SGD using Hinge loss.

	input:	1. Matrix with train examples - X 
		2. Array with train targets - y
		3. Matrix with test examples - X_test
		4. Array with test targets - y_test
		5. Number of max iterations - steps
		6. Threshold for convergenze - thresh
		7. Regularization parameter lambda - lda
		8. Batch size - batch
		9. Learning rate eta - eta

	output:	1. Array with train set loss - loss_train	
		2. Array with test set loss - loss_test
		3. Parameter array - theta
	"""

# Initialize values
   	grad, delta, abb = np.inf, np.inf, np.inf					
    	loss_test, loss_train = np.zeros(steps), np.zeros(steps)
    	m, n = X.shape						
    	i = 1
	theta = np.random.random(n)			

# Execute till convergence or max iterations
    	while  (i < steps) and (abs(delta) > thresh):

# Shuffel data
		X, y  = shuffle(X, y)			 
		X_test, y_test = shuffle(X_test, y_test)

# Calculate Hinge loss and gradient on train set
		(loss_train[i], grad) = hinge(theta, X[:batch], y[:batch], lda)

# Calculate Hinge loss and gradient on test set
		(loss_test[i], grad_test) = hinge(theta, X_test, y_test, lda)

# Calculate delta for convergence check
        	if i > 7:					
			delta = loss_test[i - 7] - loss_test[i]	

# Update theta with learning rate and gradient
        	theta = theta - eta * grad			
		i += 1
    	return (loss_train, loss_test, theta)


def main():

	"""
	Evaluation of toy problem set with mini-batch SGD using Hinge loss. 

	input values: 	1. Numpy arrays with features
			2. Numpy array with target
			3. Regularization parameter lambda
			4. Learning rate eta
			5. Batch size

	example (features = features.npy, target = target.npy, lambda = 1, eta = 0.001, batchsize = 10):

			python sgd.py features.npy target.npy 1 0.001 10

	output:		1. Time needed to converge
			2. Accuracy on test set
			3. Accuracy on train set
			4. Plot of training and test loss
	"""

# Initial Values
	lda = float(sys.argv[3])		# Reugularization parameter lambda
	eta = float(sys.argv[4])		# Learningrate eta
	batch = int(sys.argv[5])		# Batchsize
	thresh = 10 * batch			# Threshold for convergence
	steps = 10000				# Number of iterations
	start, end, acc_test, acc_train, time_needed = 0, 0, 0, 0, 0
        


# Load feature and target Data
	#X_data=loadtxt("features.txt", delimiter = ",")	
	#y_data=loadtxt("target.txt", delimiter = ",")	
	X_data = np.load(sys.argv[1])
	y_data = np.load(sys.argv[2])
	X_data, y_data  = shuffle(X_data, y_data)		# Shuffel Data 
	X_data = np.c_[np.ones(X_data.shape[0]), X_data]	# Add extra column for intercept
	


# Split Data in train and test set
	X, y = X_data[:5414] , y_data[:5414]			# Train set
	X_test, y_test = X_data[5414:] , y_data[5414:]		# Test set
	m, n = X.shape						# Some useful values
	k, l = X_test.shape

# Run mini-batch SGD and measure time
	start = time.time()
	loss_train, loss_test, theta = mbsgd(X, y, X_test, y_test, steps, thresh, lda, batch, eta)
	end = time.time()
	time_needed = end - start
	print("Time needed: %f seconds" % time_needed )

# Accuracy on test set
	pred_test = pred_hinge(X_test, theta)
	acc_test = sum(pred_test == y_test) / float(k)
	print('Accuracy on test set: %f ' % acc_test)

# Accuracy on train set
	pred_train = pred_hinge(X, theta)
	acc_train = sum(pred_train == y) / float(m)
	print('Accuracy on train set: %f' % acc_train)


# Plot test and train error
	plt.close('all')
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(loss_test[loss_test != 0])
	axarr[0].set_title('Loss on training and test set')
	axarr[0].set_ylabel("Loss test")
	axarr[1].plot(loss_train[loss_train != 0])
	axarr[1].set_xlabel("Iterations")
	axarr[1].set_ylabel("Loss training")
	plt.show()

if __name__ == "__main__":
    main()

