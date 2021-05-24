from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def false_nearest_neighbors(distance_matrix, max_length, threshold):
	"""
	Implements the false nearest neighbors algorithm
	distance matrix should be numpy array
	------
	Notes:
	Empirically observed that threshold value of 10-15 is a good pick,
	in Review of FNN paper.
	"""
	T = len(distance_matrix)
	false_nbrs = np.zeros(max_length-1)

	for l in range(1,max_length):
		
		for k in range(l,T):
			distances = np.zeros(len(range(l,T)))
			for j in range(l,T):
				distances[j - l] = \
					np.sum(np.power(distance_matrix[range(k-1,k-l-1,-1),range(j-1,j-l-1,-1)],2))
			closest_idx = np.argsort(distances)[1] # get closest neighbor
			closest_dist = distances[closest_idx]

			if distance_matrix[k,closest_idx]**2/closest_dist >= threshold**2:
				false_nbrs[l - 1] += 1 

	return false_nbrs


def noisy_quasiperiodic_test():

	ts = np.linspace(0,10,200)
	ys = np.sin(np.sqrt(2)*ts) + np.sin(np.sqrt(3)*ts) + np.sin(np.sqrt(5)*ts)

	distance_matrix = np.zeros((200,200))
	for i in range(len(ts)):
		for j in range(len(ts)):
			distance_matrix[i,j] = np.abs(ys[i] - ys[j])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(ys[:-2],ys[1:-1],ys[2:])

	plt.show()










