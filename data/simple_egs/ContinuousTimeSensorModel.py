import numpy as np 
import intervaltree as it
import scipy.spatial as sp
import SphereExample as sphere
import PlaneExample as plane
import networkx as nx
import multiprocessing as mp
from joblib import Parallel, delayed

import sys
sys.path.append('../data/')
from helper import timing

def get_sensor_lifetimes(initial_points,time, birth_rate, death_rate, manifold = 'sphere'):
	"""
	Simulates sensors lifetimes on a given space.

	See the ContinuousTimeSphereExample, or the ContinuousTimPlaneExample notebook for
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
	manifold: String
		pass "sphere" or "plane"; generates random uniform samples from one of these spaces
	Output: intervaltree
		an interval tree; each node of the tree is 
		(birth,death,coordinate) of a point.
	"""

	l1 = 1/birth_rate
	l2 = 1/death_rate

	births = [0]*initial_points #initialize 100 points
	
	current_time = 0
	intervals = it.IntervalTree()

	def sample_point():
		if manifold == 'sphere':
			return sphere.sample_uniform_sphere(1).tolist()[0]
		elif manifold == 'plane':
			return plane.sample_uniform(1).tolist()[0]
		else:
			raise Exception("spaces supported include 'sphere', 'plane' ")

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
			intervals[births.pop(0):current_time] = sample_point()

	#add the remaining nodes:
	for i in range(len(births[:-1])):
		intervals[births[i]:time] = sample_point()

	return intervals


def sample_dynamic_network(intervals, obs_times, obsfn, edge_wtsfn, manifold = 'sphere'):
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

		if manifold == 'sphere':
			hull = sp.ConvexHull(np.array(coordinates))
			node_wts.append(sphere.get_node_wts(t,hull,obsfn))
			edge_wts.append(edge_wtsfn(hull))

		elif manifold == 'plane':
			node_wts.append(plane.get_node_wts(t,np.array(coordinates),obsfn))
			edge_wts.append(edge_wtsfn(np.array(coordinates)))
		else:
			raise Exception("Please pass a handled manifold")

	return (node_wts,edge_wts,coordinate_set)

# another idea - can search for closest r' points in Eucclidean space, r' is the appropriate radius such that induced great circle on 
# the sphere has specified distance

def sample_dynamic_geometric_graph(intervals, obs_times, obsfn, manifold = 'sphere'):
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
	#num_cores = int(0.5*multiprocessing.cpu_count())

	coordinate_set = []
	node_wts = []
	edge_wts = []
	# we can maybe parallelize.
	def parallel_helper(intervals, t, obs_fn, manifold):
		points = intervals.at(t)
		coordinates = [ p[2] for p in list(points) ]
		if manifold == 'sphere':
			threshold = sphere.critical_rgg_scaling(len(points))
			node_wt = np.array([obsfn(t,np.array(p)) for p in coordinates])
			edge_wt = sphere.get_edge_wts_rgg(np.array(coordinates), threshold)
			 # convert to birthtimes
		elif manifold == 'plane':
			node_wt = plane.get_node_wts(t,np.array(coordinates),obsfn)
			edge_wt = plane.get_edge_wts_rgg(np.array(coordinates))
		else:
			return
		return (coordinates, node_wt, edge_wt)

	num_cores = mp.cpu_count() - 4
	results = Parallel(n_jobs = num_cores)(delayed(parallel_helper)(intervals,t,obsfn,manifold) for t in obs_times)

	#results = [parallel_helper(intervals,t,obsfn,manifold) for t in obs_times]

	coordinate_set = [res[0] for res in results]
	node_wts = [res[1] for res in results]
	edge_wts = [res[2] for res in results]

	return (node_wts,edge_wts,coordinate_set)


## Functions for visualization 
def visualize_dynamic_network():
	"""
	Creates a termporary folder, then creates a movie of the results
	"""
	pass


def vary_birth_death_params(T, wl, bd_rate_vals, dim_vals, manifold = 'sphere'):
	"""

	Output: array
		array of the max pers values, indexed by the birth death rate and the dimension
	Params
	------------
	T: int
		period of the observation function
	wl: float
		window length for the delay embedding
	bd_rate_vals: list of positive float:
		params for the birth death rates
	dim_vals: list of loat
		params for dims

	"""

	#Pick the window length to be half the period
	if wl is None:
		wl = T/2

	obsfn = lambda t, p: sphere.periodic_northsouth_modulated(t,p,T)
	edge_wtsfn = lambda hull_obj: sphere.get_edge_wts(hull_obj, alpha = 1.0) 


	## Vary Birth/Death rate of the sensors, and dimension
	bd_rate_test_values = bd_rate_vals
	dim_test_values = dim_vals

	mpers_results = np.zeros((len(bd_rate_test_values),len(dim_test_values)))
	top_diff_results = np.zeros((len(bd_rate_test_values),len(dim_test_values)))

	for i,bd_rate in enumerate(bd_rate_test_values):
		for j,d in enumerate(dim_test_values):
			print(bd_rate,d)
			
			# resample sensor lifetimes with different birth/death rates
			sensor_lifetimes = get_sensor_lifetimes(5*T, bd_rate, bd_rate, manifold) # this first param should be independent of analysis

			# resample dynamic network 
			tau = wl/d
			ts = np.arange(0,2*T,tau) 

			(node_wts, edge_wts, allpoints) = sample_dynamic_network(sensor_lifetimes, ts, obsfn = obsfn,
			                            edge_wtsfn = edge_wtsfn)

			PDs = sphere.apply_pipeline(node_wts,edge_wts, d = d, tau = 1, lamda=1, phi=sphere.linear_phi_fn) # get the PDs
			res = (sphere.get_maximum_persistence(PDs)[1], \
					sphere.get_top_diff_persistence(PDs)[1], \
					sphere.get_num_features(PDs)[1])
			print(res)

			mpers_results[i,j] = res[0]
			top_diff_results[i,j] = res[1]

	return mpers_results


