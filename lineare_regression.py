import numpy as np
import matplotlib.pyplot as plt



def scr_theta_estimation(x, y, steps, alpha):
	"""input :  data, steps and learning rate
	   output : theta estimation
	   function : calculate theta estimation
	"""
	theta = [0,0]
	for idStep in range(steps):
		for i in range(len(x)):
			for idTheta in range(2):
				e = y[i] - theta[idTheta] * x[i][idTheta]
				theta[idTheta] = theta[idTheta] + alpha * (2 * e * x[i][idTheta])

		for idTheta in range(2):
			e = 0
			for idX in range(len(x)):
				e = e + pow(y[idX] - theta[idTheta] * x[idX][idTheta], 2)
			if e < 0.001:
				break
	return theta



###########################
if __name__ == '__main__':

	#Generate data
	mean = np.array([0,0])
	P = np.array([[1, 1], [-1, 1]]) # kind of rotation matrix, to bend my distribution
	cov = np.dot(np.dot(P, np.array([[0.1,0],[0,1]])), np.linalg.inv(P))

	x, y = np.random.multivariate_normal(mean, cov, 100).T
	plt.axis([-3,3,-3,3])
	plt.scatter(x, y)
	x1 = [1 for i in range(len(x))]
	X = np.column_stack((x,x1))
	#print(X)

	##estimation of theta
	alpha = 0.0002
	steps = 5000
	theta = scr_theta_estimation(X, y, steps, alpha)
	print(theta)

	##prediction
	abcisses = np.linspace(-3, 3, 500)
	predict_y = theta[0] * abcisses + theta[1]
	plt.scatter(predict_y, abcisses, color = "red")
	plt.show()
