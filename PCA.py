import numpy as np
import matplotlib.pyplot as plt


def pca(X):
	#We need the covariance Matrix, here it is : 1/n * t(X) * X
	covX = 1/len(X) * np.dot(np.transpose(X), X)
	#The objective is to change the baseline, so we want to have the eigen vector. The eigen vector exist because covX can be diagonalize.
	vecPropre = np.linalg.eig(covX)[1]
	newX = np.dot(X,vecPropre)
	return newX

#Generate data
mean = np.array([0,0])
P = np.array([[1, 1], [-1, 1]]) # kind of rotation matrix, to bend my distribution
cov = np.dot(np.dot(P, np.array([[0.1,0],[0,1]])), np.linalg.inv(P))

print(cov)

x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.axis([-3,3,-3,3])
plt.scatter(x, y)
plt.show()

#We concatenate x and y, to have a sample of points
X = np.column_stack((x,y))

newX = pca(X)

plt.axis([-3,3,-3,3])
plt.scatter(newX[:,0], newX[:,1])
plt.show()
