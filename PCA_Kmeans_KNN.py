import numpy as np
import matplotlib.pyplot as plt
import Kmeans



def pca(X):
	"""input : X, all the point
       output : X projected on the principal components
	   function : find the principal components of X
	"""
	#We need the covariance Matrix, here it is : 1/n * t(X) * X
	covX = 1/len(X) * np.dot(np.transpose(X), X)
	#The objective is to change the baseline, so we want to have the eigen vector. The eigen vector exist because covX can be diagonalize.
	vecPropre = np.linalg.eig(covX)[1]
	newX = np.dot(X,vecPropre)
	return newX


def distance(pointA, allPoint):
	""" input : pointA with unknow classification, allPoint of data set
		output : array with all distance
		function : find distance with all point to find KNN
	"""
	return [np.linalg.norm(pointA - m) for m in allPoint]


def classification(K, distance, classePoint, numberClasse):
	"""
	input : K nearest neighbour, distance between A and other point, cluster of each point, number of cluster
	output : classification of pointA and the KNN
	function : find the cluster of A
	"""
	kNn = []
	classeUnknow = [0 for i in range(numberClasse)]
	if K < len(distance):
		for i in range(K):
			kNn.append(np.argmin(distance)) #find the K nearest neighbour
			distance[np.argmin(distance)] = 10000000

		print(kNn)

		for i in range(len(kNn)):
			print("classe k nearest : ", classePoint[kNn[i]])
			classeUnknow[classePoint[kNn[i]]] += 1 #check the class of K nearest neighbour


	return classeUnknow, kNn



###########################
if __name__ == '__main__':

#GENERATE DATA SET
	mean = []
	mean.append(np.array([1,-1]))
	mean.append(np.array([-1,1]))
	cov = np.array([[0.1,0],[0,0.1]])
	X = np.concatenate([np.random.multivariate_normal(m, cov, 100) for m in mean])
	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], c="blue")
	plt.show()

	X = pca(X)
	meanA = []
	meanA.append(np.array([0,0]))
	A = np.concatenate([np.random.multivariate_normal(m, cov, 1) for m in meanA])
	numberClasse = 2
	curKMeans, cluster = Kmeans.initializeCluster(X, numberClasse)

	prev_means = np.array([1])
	cluster, curKMeans = Kmeans.kmeans(X, prev_means, curKMeans, cluster)

	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], c=cluster)
	plt.scatter(A[:,0], A[:,1], c="yellow")
	plt.scatter(curKMeans[:,0], curKMeans[:,1], color="red", s=20)
	plt.show()

#Find cluster of A
	k = 3
	kNn = []
	classA, kNn = classification(k, distance(A, X), cluster, numberClasse)
	print("La classe de A vaut : ", np.argmax(classA))
	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], c=cluster)
	for i in range(len(kNn)):
		plt.scatter(X[kNn[i],0], X[kNn[i],1], c="yellow")
	plt.scatter(A[:,0], A[:,1], c="yellow")
	plt.scatter(curKMeans[:,0], curKMeans[:,1], color="red", s=20)
	plt.show()
