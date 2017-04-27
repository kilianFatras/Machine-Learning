import numpy as np
import matplotlib.pyplot as plt

def nearest_mean(point, means):
	return np.argmin([np.linalg.norm(point - m) for m in means]) # return the index of the mean in "means" array

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

CurKMeans = np.array([X[i] for i in range(K)])
cluster = np.array([nearest_mean(p, CurKMeans) for p in X])

plt.axis([-3,3,-3,3])
plt.scatter(X[:,0], X[:,1], c=cluster)
plt.scatter(CurKMeans[:,0], CurKMeans[:,1], color="red", s=20)
plt.show()

#LOOP
prev_means = np.array([1])
while ((prev_means != CurKMeans).any()):
	prev_means = CurKMeans.copy()
	CurKMeans = np.array([np.mean(X[cluster == l], axis=0) for l in range(len(CurKMeans))])
	cluster = np.array([nearest_mean(p, CurKMeans) for p in X])

cluster = np.array([nearest_mean(p, CurKMeans) for p in X])

plt.axis([-3,3,-3,3])
plt.scatter(X[:,0], X[:,1], c=cluster)
plt.scatter(CurKMeans[:,0], CurKMeans[:,1], color="red", s=20)
plt.show()
