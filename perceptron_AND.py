"""Neural network learns the AND logical function"""


import numpy as np
import random



def sigmoid(x):
	return 1./(1+np.exp(-x))

class perceptron():
	def __init__(self):
		self.w = [random.random() for i in range(2)]
		self.b = random.random()

	def forward(self, x):
		return sigmoid(np.dot(np.transpose(self.w), x) + self.b)

	def backprop(self, x, y, lr):
		y_predict = self.forward(x)
		err = y_predict - y
		self.b -= lr * err * y_predict * (1. - y_predict) #optimization bias
		for id_w in range(2):
			self.w[id_w] -= lr * err * y_predict * (1. - y_predict) * x[id_w] #optimization weights


if __name__ == '__main__':

	nb_iter = 50000
	X = []
	Y = []
	for i in range(nb_iter):
	    x1 = random.randint(0,1)
	    x2 = random.randint(0,1)
	    X.append([x1,x2])
	    Y.append([x1 & x2])
	X = np.array(X, dtype=float)
	Y = np.array(Y, dtype=float)

	model = perceptron()
	for i in range(50000):
		model.backprop(X[i], Y[i], 0.01)

	x = [1,1]
	print("Weights : ", model.w)
	print("Bias : ", model.b)
	print()
	print("Result : ", np.round(model.forward(x)[0]))
	print("Probability : ", model.forward(x)[0])
