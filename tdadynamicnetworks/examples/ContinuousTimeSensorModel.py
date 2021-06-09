import numpy as np 
import scipy.spatial as sp
from . import SphereExample as sphere
from . import PlaneExample as plane
import multiprocessing as mp
from joblib import Parallel, delayed
from collections import deque

def get_alive_sensor_locs(intervals, t):
	"""
	Return location of sensors that are alive at a particular time

	Parameters
	----------
	intervals: tuple
		(ndarray(N, 2) of birth deaths, ndarray(N, d) of coordinates) of a each sensor.
	t: float
		Time at which to query sensors
	
	Returns
	-------
	ndarray(M <= N, d)
		Coordinates of found sensors
	"""
	b = intervals[0][:, 0]
	d = intervals[0][:, 1]
	return intervals[1][(t >= b)*(t <= d), :]


def get_sensor_lifetimes(initial_points,time, birth_rate, death_rate, domain_lengths = (1,1), manifold = 'sphere', seed = 17):
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
	Output: tuple
		(ndarray(N, 2) of birth deaths, ndarray(N, d) of coordinates) of a each sensor.
	"""
	np.random.seed(seed)

	l1 = 1/birth_rate
	l2 = 1/death_rate

	births = deque()
	for i in range(initial_points):
		births.append(0)
	
	current_time = 0
	intervals = []
	points = []

	def sample_point():
		if manifold == 'sphere':
			return sphere.sample_uniform_sphere(1).tolist()[0]
		elif manifold == 'plane':
			return plane.sample_uniform(1, domain_lengths[0], domain_lengths[1]).tolist()[0]
		elif manifold == 'circle':
			return np.random.uniform()
		else:
			raise Exception("spaces supported include 'sphere', 'plane','circle' ")

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
			intervals.append([births.popleft(), current_time])
			points.append(sample_point())

	#add the remaining nodes:
	while len(births) > 0:
		intervals.append([births.popleft(), current_time])
		points.append(sample_point())
	intervals = np.array(intervals)
	points = np.array(points)
	return (intervals, points)


def get_fixed_sensors(initial_points,time, domain_lengths = (1,1), manifold = 'sphere', seed = 17):
	"""
	Simulates sensors lifetimes on a given space.

	See the ContinuousTimeSphereExample, or the ContinuousTimPlaneExample notebook for
	details on implementation.

	
	Parameters
	-------------------------------------
	initial_points: int
		Number of points to sample
	time: float
		simulate the queueing model up to this time
	manifold: String
		pass "sphere" or "plane"; generates random uniform samples from one of these spaces
	Output: intervaltree
		an interval tree; each node of the tree is 
		(birth,death,coordinate) of a point.
	"""
	def sample_point():
		if manifold == 'sphere':
			return sphere.sample_uniform_sphere(1).tolist()[0]
		elif manifold == 'plane':
			return plane.sample_uniform(1, domain_lengths[0], domain_lengths[1]).tolist()[0]
		elif manifold == 'circle':
			return np.random.uniform()
		else:
			raise Exception("spaces supported include 'sphere', 'plane','circle' ")
	intervals = []
	points = []
	for i in range(initial_points):
		intervals.append([0, time])
		points.append(sample_point())
	intervals = np.array(intervals)
	points = np.array(points)
	return (intervals, points)


def sample_dynamic_network(intervals, obs_times, obsfn, edge_wtsfn, manifold = 'sphere'):
	"""
	Given set of observations, creates the dynamic network at those times
	given the birth/death times of the sensors. 

	Parameters
	----------------------------
	Input:
	intervals: tuple
		(ndarray(N, 2) of birth deaths, ndarray(N, d) of coordinates) of a each sensor.
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
		coordinates = get_alive_sensor_locs(intervals, t)
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


def sample_dynamic_geometric_graph(intervals, obs_times, obsfn, manifold = 'sphere', rescale_node_weight = None, verbose=False):
	"""
	Given set of observations, creates the dynamic network at those times
	given the birth/death times of the sensors. 

	Parameters
	----------------------------
	Input:
	intervals: tuple
		(ndarray(N, 2) of birth deaths, ndarray(N, d) of coordinates) of a each sensor.
	obs_times: list of floats
		list of times at which to sample obs function
	obsfn:
		function from which to sample
	Output: tuple
		(node_wts,edge_wts,all_points)
		node_wts: list of node wts at each time index
	verbose: boolean
		Whether to print timing information
	"""
	### query tree at each timestep ###
	#num_cores = int(0.5*multiprocessing.cpu_count())
	import time

	coordinate_set = []
	node_wts = []
	edges = []
	# we can maybe parallelize.
	def parallel_helper(intervals, t, obsfn, manifold, rescale_node_weight):
		tic = time.time()
		coordinates = get_alive_sensor_locs(intervals, t)
		N = coordinates.shape[0]
		if verbose:
			print("Elapsed time intervals: {}".format(time.time()-tic))
		if manifold == 'sphere':
			threshold = sphere.critical_rgg_scaling(N)
			node_wt = np.array([obsfn(t,np.array(p)) for p in coordinates])
			edge_wt = sphere.get_edge_wts_rgg(np.array(coordinates), threshold) ## TODO

		elif manifold == 'plane':
			threshold = plane.supercritical_rgg_scaling(N)
			tic = time.time()
			node_wt = plane.get_node_wts(t,np.array(coordinates),obsfn)
			if verbose:
				print("Elapsed Time Nodes: {}".format(time.time()-tic))
			tic = time.time()
			edge = plane.get_edges_rgg(np.array(coordinates), threshold)
			if verbose:
				print("Elapsed Time Edges: {}".format(time.time()-tic))

		elif manifold == 'torus':
			threshold = plane.critical_rgg_scaling(N)
			node_wt = plane.get_node_wts(t,np.array(coordinates),obsfn)
			edge_wt = plane.get_edge_wts_rgg_torus(np.array(coordinates), threshold) ## TODO

		elif manifold == 'circle':
			threshold = plane.supercritical_rgg_scaling_circle(N)
			node_wt = plane.get_node_wts(t,np.array(coordinates),obsfn)
			edge_wt = plane.get_edge_wts_rgg_circle(np.array(coordinates), threshold) ## TODO	

		elif manifold == 'interval':
			threshold = plane.supercritical_rgg_scaling_circle(N)
			node_wt = plane.get_node_wts(t,np.array(coordinates),obsfn)
			edge_wt = plane.get_edge_wts_rgg_interval(np.array(coordinates), threshold) ## TODO	
		else:
			return

		if rescale_node_weight:
			node_wt = node_wt*rescale_node_weight/np.max(np.abs(node_wt))
		return (coordinates, node_wt, edge)

	num_cores = mp.cpu_count() - 4
	#results = Parallel(n_jobs = num_cores)(delayed(parallel_helper)(intervals,t,obsfn,manifold,rescale_node_weight) for t in obs_times)
	results = [parallel_helper(intervals,t,obsfn,manifold,rescale_node_weight) for t in obs_times]

	#results = [parallel_helper(intervals,t,obsfn,manifold) for t in obs_times]

	coordinate_set = [res[0] for res in results]
	node_wts = [res[1] for res in results]
	edges = [res[2] for res in results]

	return (node_wts,edges,coordinate_set)



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


