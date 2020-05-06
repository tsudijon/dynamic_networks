import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix

def sample_uniform_sphere(N, seed=17):
    np.random.seed(seed)
    cds = np.random.normal(0, 1, (N, 3))
    normalized_cds = cds / np.reshape(np.sqrt(np.sum(cds ** 2, axis=1)), (N, 1))
    return normalized_cds


# cds should be an array of size (3,)
def cartesian_to_spherical(cds):
    (x, y, z) = cds
    rho = np.sqrt(np.sum(cds ** 2))
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arccos(x / r)
    phi = np.arcsin(r / rho)

    return (rho, phi, theta)


def cartesian_to_sphere_distance(cds1, cds2):
    (rho, lat1, lon1) = cartesian_to_spherical(cds1)
    (rho, lat2, lon2) = cartesian_to_spherical(cds2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat2) * np.cos(lat1) * np.sin(dlon / 2) ** 2)
    c = np.arcsin(a)

    return 2 * rho * c

def get_spherical_distance_matrix(points):
    """
    
    :param points: Nx3 array
    :return:
    """
    n = points.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = cartesian_to_sphere_distance(points[i], points[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances

def get_rgg_adjacency_matrix(points, epsilon, distance_func = 'Sphere'):
    """
    Get the adjaency amtrix of an rgg graph given the points of the graph.
    :param points:
    :param epsilon:
    :return:
    """
    if distance_func == 'Sphere':
        dists = get_spherical_distance_matrix(points)


    matrix = dists[dists <= epsilon]
    matrix[matrix != 0] = 1

    return matrix

def get_rgg_birthtime_matrix(points, epsilon, distance_func = 'Sphere'):
    """
    Get the adjaency amtrix of an rgg graph given the points of the graph.
    :param points:
    :param epsilon:
    :return:
    """
    if distance_func == 'Sphere':
        dists = get_spherical_distance_matrix(points)
    elif distance_func == 'Euclidean':
        dists = distance_matrix(points, points)
    else:
        print("Distance type not supported.")
        return None

    dists = np.where(dists < epsilon, -np.inf, np.inf) #if value is -infinity, then edge will be born at max of vertex birth times.

    return sparse.coo_matrix(dists)


def get_node_wts(t, points, obsfn):
    f = lambda x: obsfn(t,x)
    node_wts = np.apply_along_axis(f, axis=1, arr=points)
    return np.array(node_wts)

# Sphere Funkywunks~~
# node weights are sampled from a periodic function on the sphere. sin(t + 2*pi*z)?
def periodic_northsouth_modulated(t,cds, T):
    """
    Parameters
    ----------
    t: float
        Time index
    cds: ndarray(N, 3)
        Cartesion coordinates of sphere points
    T: float
        Period of a cycle
    """
    rho, phi, theta = cartesian_to_spherical(cds)
    return 3 + 2*np.cos(2*np.pi*t/T + 2*phi)

def sph_harm_modulated(t,cds, T, m, n):
    """
    A modulated spherical harmonic
    Parameters
    ----------
    t: float
        Time index
    cds: ndarray(N, 3)
        Cartesion coordinates of sphere points
    T: float
        Period of a cycle
    m: int
        m parameter for spherical harmonics
    n: int
        n parameter for spherical harmonics
    """
    from scipy.special import sph_harm
    rho, phi, theta = cartesian_to_spherical(cds)
    return np.sin(2*np.pi*t/T)*np.real(sph_harm(n, m, theta, phi))