import numpy as np
import matplotlib.pyplot as plt



def nearest_mean(point, means):
	return np.argmin([np.linalg.norm(point - m) for m in means]) # return the index of the mean in "means" array

def initializeCluster(X, K):
	curKMeans = np.array([X[i] for i in range(K)])
	cluster = np.array([nearest_mean(p, curKMeans) for p in X])
	return curKMeans, cluster

def kmeans(X, prev_means, curKMeans, cluster):
	while ((prev_means != curKMeans).any()):
		prev_means = curKMeans.copy()
		curKMeans = np.array([np.mean(X[cluster == l], axis=0) for l in range(len(curKMeans))])
		cluster = np.array([nearest_mean(p, curKMeans) for p in X])
	return cluster, curKMeans


####################
if __name__ == '__main__':

	#GENERATE DATA SET
	mean = []
	mean.append(np.array([0,0]))
	cov = np.array([[0.1,0],[0,0.1]])
	X = np.concatenate([np.random.multivariate_normal(m, cov, 250) for m in mean])
	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], color="blue")
	plt.show()

	#INITIALIZATION
	print("Choose K : ")
	K = int(input())

	curKMeans, cluster = initializeCluster(X)

	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], c=cluster)
	plt.scatter(curKMeans[:,0], curKMeans[:,1], color="red", s=20)
	plt.show()

	#LOOP
	prev_means = np.array([1])
	cluster, curKMeans = kmeans(prev_means, curKMeans, cluster)

	plt.axis([-3,3,-3,3])
	plt.scatter(X[:,0], X[:,1], c=cluster)
	plt.scatter(curKMeans[:,0], curKMeans[:,1], color="red", s=20)
	plt.show()
