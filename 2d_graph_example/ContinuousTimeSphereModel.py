import numpy as np 
import intervaltree as it
import scipy.spatial as sp

def get_sensor_lifetimes(time, birth_rate, death_rate):
	'''
	A shoddy implementation of a queueing model, with exponentially distributed processing time and
	exponentially distributed growth.
	'''
	l1 = 1/birth_rate
	l2 = 1/death_rate

	births = [0]*100 #initialize 100 points
	
	current_time = 0
	intervals = it.IntervalTree()

	while current_time < time:

		# if there are no current points
		if len(births) == 0:
			current_time += np.random.exponential(l2)
			births.append(current_time)
			continue

		current_time += np.random.exponential(1/(birth_rate + death_rate))

		# add a new point
		if np.random.rand() < l1/(l1 + l2):
			births.append(current_time)
		else:
			intervals[births.pop(0):current_time] = sample_uniform_sphere(1).tolist()[0]

	#add the remaining nodes:
	for i in range(len(births[:-1])):
		intervals[births[i]:time] = sample_uniform_sphere(1).tolist()[0]

	return intervals

def create_dynamic_networks(lambda1,lambda2, obs_times, observation_function, edge_wtsfn):
	'''
	Can query the interval tree below for more information about the points
	-----------------------------------------------------------------------
	Input:
	lambda1: birth rate (we expect )
	lambda2: death rate (length of a lifetime of a node)
	obs_times - list of times at which to sample obs function
	observation_function - function from which to sample
	edge_wtsfn - function applied to edge wts
	'''

	### create the interval tree ###
	intervals = get_sensor_lifetimes(max(obs_times), lambda1, lambda2)

	### query tree at each timestep ###
	coordinate_set = []
	for t in obs_times:
		points = intervals.at(t)
		coordinates = [ p[2] for p in list(points) ]

		coordinate_set.append(coordinates)

		hull = sp.ConvexHull(coordinates)
		node_wts.append(get_node_wts(t,hull,observation_function))
		edge_wts.append(edge_wtsfn(hull))

	return (node_wts,edge_wts,point_set)



def sample_uniform_sphere(N):
	cds = np.random.normal(0,1,(N,3))
	normalized_cds = cds/np.reshape(np.sqrt(np.sum(cds**2,axis = 1)),(N,1))
	return normalized_cds

def get_node_wts(t,hull_obj, obsfn):
    hull = hull_obj
    node_wts = [obsfn(t,p) for p in hull.points]
    return np.array(node_wts)

def get_edge_wts(hull_obj, alpha = 1.0):
	"""
	map edges to their distances
	use this to create a dynamic network, edges are these distances.
	Returns
	-------
	edges: scipy.sparse(N, N)
	    A sparse matrix with the edge weights
	alpha: float
	    Amount by which to weight distances
	"""
	hull = hull_obj
	edges = simplex_list_to_edge_list(hull.simplices)

	v = hull.points
	ds = [cartesian_to_sphere_distance(v[e[0],:], v[e[1],:]) for e in edges]
	ds = alpha*np.array(ds + ds)

	e0 = np.array([e[0] for e in edges] + [e[1] for e in edges])
	e1 = np.array([e[1] for e in edges] + [e[0] for e in edges])

	return sparse.coo_matrix((ds, (e0, e1)), shape=(len(v), len(v)))


