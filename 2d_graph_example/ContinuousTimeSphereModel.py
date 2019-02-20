import numpy as np 
import intervaltree as it
import scipy.spatial as sp
import SphereExample as sphere

def get_sensor_lifetimes(time, birth_rate, death_rate):
	"""
	Simulates sensors lifetimes on the sphere.

	See the ContinuousTimeSphereExample notebook for
	details on implementation.

	
	Parameters
	-------------------------------------
	time: float
		simulate the queueing model up to this time
	birth_rate: float > 0
		birth rate (we expect these many to be born
		in a unit time interval)
	death_rate: float > 0
		death rate
	Output: intervaltree
		an interval tree; each node of the tree is 
		(birth,death,coordinate) of a point.
	"""

	l1 = 1/birth_rate
	l2 = 1/death_rate

	births = [0]*100 #initialize 100 points
	
	current_time = 0
	intervals = it.IntervalTree()

	while current_time < time:

		# if there are no points in the queue 
		if len(births) == 0:
			current_time += np.random.exponential(l2)
			births.append(current_time)
			continue

		current_time += np.random.exponential(1/(birth_rate + death_rate))

		# add a new point
		if np.random.rand() < l1/(l1 + l2):
			births.append(current_time)
		else:
			intervals[births.pop(0):current_time] = \
						sphere.sample_uniform_sphere(1).tolist()[0]

	#add the remaining nodes:
	for i in range(len(births[:-1])):
		intervals[births[i]:time] = sphere.sample_uniform_sphere(1).tolist()[0]

	return intervals


def sample_dynamic_network(intervals, obs_times, obsfn, edge_wtsfn):
	"""
	Given set of observations, creates the dynamic network at those times
	given the birth/death times of the sensors. 

	Parameters
	----------------------------
	Input:
	intervals: intervaltree
		output of get_sensor_lifetimes
	obs_times: list of floats
		list of times at which to sample obs function
	obsfn:
		function from which to sample
	edge_wtsfn:
		function applied to edge wts
	Output: tuple
		(node_wts,edge_wts,all_points)
		node_wts: list of node wts at each time index
	"""


	### query tree at each timestep ###
	coordinate_set = []
	node_wts = []
	edge_wts = []
	for t in obs_times:
		points = intervals.at(t)
		coordinates = [ p[2] for p in list(points) ]

		coordinate_set.append(coordinates)

		hull = sp.ConvexHull(np.array(coordinates))
		node_wts.append(sphere.get_node_wts(t,hull,obsfn))
		edge_wts.append(edge_wtsfn(hull))

	return (node_wts,edge_wts,coordinate_set)

## Functions for visualization 

def visualize_dynamic_network():
	"""
	Creates a termporary folder, then creates a movie of the results
	"""
	pass

