import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class NueralNetwork:
	def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):

		self.input_layer = input_size
		self.hidden_layer = hidden_size
		self.output_layer = output_size
    	
		if activation == 'sigmoid':
			self.activation = self.sigmoid
		else:
			raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")

		self.params =  self.initialize()

	# Activiaation function
	def sigmoid(self, x, derivative=False):
		if derivative:
			return (np.exp(-x))/((np.exp(-x)+1)**2)
		return(1/(1 + np.exp(-x)))
	
	def initialize(self):
		params = {
            "W1": np.random.randn(self.hidden_layer, self.input_layer) * np.sqrt(1./self.input_layer),
            "b1": np.zeros((self.hidden_layer, 1)) * np.sqrt(1./self.input_layer),
            "W2": np.random.randn(self.output_layer, self.hidden_layer) * np.sqrt(1./self.hidden_layer),
            "b2": np.zeros((self.output_layer, 1)) * np.sqrt(1./self.hidden_layer)
        }
		return params
	
	# input Y true, and predicted y_hat between 0 and 1
	def compute_loss(self, y, y_hat):		
		L = -np.multiply(y, np.log(y_hat)) - np.multiply(1-y, np.log(1-y_hat))

		return np.mean(L)

	def f_forward(self, x):

		z1 = np.dot(x, self.params['W1'].T)  + self.params["b1"].T
		a1 = self.activation(z1)

		z2 = np.dot(a1, self.params['W2'].T) + self.params["b2"]
		a2 = self.activation(z2)
		
		return [z1,a1, z2, a2]
	
	def back_propagate(self, x, y, output, alpha):
		
		a1 = output[1]
		a2 = output[3]

		w1 = self.params["W1"]
		w2 = self.params['W2']

		current_batch_size = y.shape[0]


		# error in output layer
		y = y.reshape(-1, 1)
		d2 =a2-y
	
		d1 = np.multiply(np.dot(d2, w2), np.multiply(a1, 1-a1))
		w1_adj = (1./current_batch_size)*np.dot(d1.T , x)
		w2_adj = (1./current_batch_size)*np.dot(d2.T, a1)
		
		b1_adj = (1/current_batch_size)*np.sum(d1, axis=0, keepdims=True).T  # shape: (5, 1)
		b2_adj = (1/current_batch_size)*np.sum(d2, axis=0, keepdims=True).T

		# Updating parameters
		self.params["W1"] = w1-(alpha*(w1_adj))
		self.params['W2'] = w2-(alpha*(w2_adj))

		# # Bias updates (gradient descent)
		self.params["b1"] = self.params["b1"] - (alpha * b1_adj)
		self.params["b2"] = self.params["b2"] - (alpha * b2_adj)
				

	def train_model(self,X_train, Y_train, X_test, Y_test, epoch=10, alpha=0.01, batch_size=64):
		train_acc = []
		train_loss = []

		test_acc, test_loss = [],[]

		num_batches = -(-X_train.shape[0] // batch_size)

		for i in range(epoch):
			permutation = np.random.permutation(X_train.shape[0])
			x_train_shuffled = X_train[permutation]
			y_train_shuffled = Y_train[permutation]

			for j in range(num_batches):
				begin = j * batch_size
				end = min(begin + batch_size, X_train.shape[0]-1)
				x = x_train_shuffled[begin:end]
				y = y_train_shuffled[begin:end]

				output = self.f_forward(x)

				self.back_propagate(x, y, output, alpha)

			
			###  ========================== TESTING PORTION =================== ###
			# Training Test
			acc, _, _, _, loss = self.test_model(x_train_shuffled, y_train_shuffled)
			train_acc.append(acc)
			train_loss.append(loss)

			# Testing Test
			acc2, _, _, _, loss2 = self.test_model(X_test, Y_test)
			test_acc.append(acc2)
			test_loss.append(loss2)
			print(f"Epoch {i + 1:>2} | Train Acc: {acc:.4f} | Train Loss: {loss:.4f} || Test Acc: {acc2:.4f} | Test Loss: {loss2:.4f}")
			###  ========================== TESTING PORTION ENDS =================== ###
		
		return train_acc, train_loss, test_acc, test_loss

	# Input X ndarray --- output
	def predict(self, X):
		output = self.f_forward(X)
		y_pred = output[3].flatten()
		return y_pred

	def test_model(self, X, Y):
		output = self.f_forward(X)

		y_true = Y.flatten()
		y_pred = output[3].flatten()

		# Loss calculated using binary entopy 
		loss = self.compute_loss(y_true, y_pred)

		# Finding the labels
		labels = (output[3] >= 0.5).astype(int)
		y_pred = labels.flatten()

		# Calculate Accuracy, precisioin, recall, and F1
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, zero_division=1)
		recall = recall_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)

		return accuracy, precision, recall, f1, loss

