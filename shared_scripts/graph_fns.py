''' February 19, 2018
BP: Functions to modify graphs'''

from __future__ import division
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def shifted_arctan(x):
    ''' a possible phi function'''
    return 1 - np.arctan(x)


def shifted_inv_fn(x):
    ''' a possible phi function'''
    return 1 / (x + 1)


def softplus(x):
    ''' a possible phi function'''
    return np.log(1 + np.exp(x))


def relu(x):
    ''' a possible phi function'''
    return np.max(0, x)


def exp_decay(x):
    ''' possible phi fn'''
    return np.exp(-x)


# pass node weights and edge weights thru the phi function
def weight_fn(node_wts, edge_wts, lamda, phi='softplus'):
    if phi == 'softplus':
        phi_fn = lambda x: softplus(x)
    elif phi == 'relu':
        phi_fn = lambda x: relu(x)
    else:  # custom phi fn
        phi_fn = phi

    assert len(node_wts) == len(edge_wts), 'Unequal number of edge wts and node wts'

    phi_node_wts = np.array(list(map(phi_fn, lamda * node_wts)))
    phi_edge_wts = np.array(list(map(phi_fn, edge_wts)))
    return phi_node_wts, phi_edge_wts


def getTorusAdj(N):
    """
    Return the sparse adjacency matrix, which indexes
    the vertices in row major order
    :param N: Making an NxN torus grid
    :returns (A: An N^2xN^2 sparse adjacency matrix, \
              pos2idx: An NxN matrix indexing the vertices,\
              idx2pos: An N^2 x 2 matrix of positions of each index
    """
    # Create an N x N grid
    pos2idx = np.arange(N * N)
    pos2idx = np.reshape(pos2idx, (N, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(N))
    idx2pos = np.zeros((N * N, 2))
    I = []
    J = []
    for i in range(N):
        for j in range(N):
            i1 = YPos[i, j]
            i2 = (i1 + 1) % N
            j1 = XPos[i, j]
            j2 = (j1 + 1) % N
            a = pos2idx[i1, j1]
            b = pos2idx[i2, j1]
            c = pos2idx[i1, j2]
            I += [a, b, a, c]
            J += [b, a, c, a]
            idx2pos[pos2idx[i1, j1], :] = [i1, j1]
    I = np.array(I)
    J = np.array(J)
    V = np.ones(len(I))
    A = sparse.coo_matrix((V, (I, J)), shape=(N * N, N * N)).tocsr()
    return (A, pos2idx, idx2pos)


def draw2DGraph(A, idx2pos, vals, drawText=False):
    """
    Render the graph and its edges
    :param A: A sparse adjacency matrix
    :param idx2pos: An NVerticesx2 array of vertex positions
    :param vals: An NVertices-length array of scalar function values\
        assumed to be between 0 and 1
    :param drawText: If true, label the vertices by their index
    """
    c = plt.get_cmap('gray')
    C = c(np.array(np.round(vals * 255.0), dtype=np.int32))
    C = C[:, 0:3]
    # First draw vertices
    plt.scatter(idx2pos[:, 0], idx2pos[:, 1], 200, c=C)
    if drawText:
        for i in range(A.shape[0]):
            pos = idx2pos[i, :]
            plt.text(pos[0] + 0.15, pos[1] + 0.15, "%i" % i)
    # Now draw edges
    Ac = A.tocoo()
    I = Ac.row.tolist()
    J = Ac.col.tolist()
    for idx1, idx2 in zip(I, J):
        pos1 = idx2pos[idx1]
        pos2 = idx2pos[idx2]
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k')


def interpolateStates(V1, V2, NFrames):
    """
    Linearly interpolate between two scalar functions V1 and V2
    """
    N = V1.size
    X = np.zeros((NFrames, N))
    ts = np.linspace(0, 1, NFrames)
    for i, t in enumerate(ts):
        X[i, :] = V1 * (1 - t) + V2 * t
    return X


def getRandomPosIdxs(K, pos2idx):
    """
    Choose a random 2x2 location on a KxK torus
    """
    # Pick a random start position for the square
    pos = np.random.random_integers(0, K - 1, 2)
    pos = [[pos[0], pos[1]],
            [pos[0] + 1, pos[1]],
            [pos[0] + 1, pos[1] + 1],
            [pos[0], pos[1] + 1]]
    pos = np.array(pos)
    pos = np.mod(pos, K)
    pos = pos.tolist()
    idxs = [pos2idx[i1, i2] for [i1, i2] in pos]
    return np.array(idxs)


def getBlinkingVideo(K, NPeriods, TransLen):
    """
    Make a video of four dots blinking down to 4, 3, 2, 1, 0 and then back
    at random positions on a KxK torus graph
    :param K: Make a KxK torus
    :param NPeriods: Number of blinking periods to go through
    :param TransLen: Length of transition from one state to the next.\
          Period will be this total length times 9
    :returns {'XRet': A (TransLen*9*NPeriodx) x (KxK) array of scalar values\
                    at the vertices of the torus at each point in time\
              'idx2pos':idx2pos, 'A':Adjacency matrix}
    """
    XRet = np.array([])
    (A, pos2idx, idx2pos) = getTorusAdj(K)
    N = A.shape[0]
    idxs = getRandomPosIdxs(K, pos2idx)
    for T in range(NPeriods):
        # First count down
        k = len(idxs)
        while k > 0:
            V1 = np.zeros(N)
            V1[idxs[0:k]] = 1.0
            V2 = np.zeros(N)
            V2[idxs[0:k - 1]] = 1.0
            thisX = interpolateStates(V1, V2, TransLen)
            if XRet.size == 0:
                XRet = thisX
            else:
                XRet = np.concatenate((XRet, thisX), 0)
            k -= 1
        # Now count up at a different place
        idxs = getRandomPosIdxs(K, pos2idx)
        while k < len(idxs):
            V1 = np.zeros(N)
            V1[idxs[0:k]] = 1.0
            V2 = np.zeros(N)
            V2[idxs[0:k + 1]] = 1.0
            thisX = interpolateStates(V1, V2, TransLen)
            XRet = np.concatenate((XRet, thisX), 0)
            k += 1
    return {'XRet': XRet, 'A': A, 'idx2pos': idx2pos, 'pos2idx': pos2idx}


