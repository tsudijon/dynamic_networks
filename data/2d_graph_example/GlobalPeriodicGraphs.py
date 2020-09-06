import sys
import scipy.sparse as sparse
import numpy as np
import scipy.io as sio
import scipy.interpolate
import matplotlib.pyplot as plt


def getTorusAdj(N):
    """
    Return the sparse adjacency matrix, which indexes
    the vertices in row major order
    :param N: Making an NxN torus grid
    :returns (A: An N^2xN^2 sparse adjacency matrix, \
              pos2idx: An NxN matrix indexing the vertices,\
              idx2pos: An N^2 x 2 matrix of positions of each index
    """
    #Create an N x N grid
    pos2idx = np.arange(N*N)
    pos2idx = np.reshape(pos2idx, (N, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(N))
    idx2pos = np.zeros((N*N, 2))
    I = []
    J = []
    for i in range(N):
        for j in range(N):
            i1 = YPos[i, j]
            i2 = (i1+1)%N
            j1 = XPos[i, j]
            j2 = (j1+1)%N
            a = pos2idx[i1, j1]
            b = pos2idx[i2, j1]
            c = pos2idx[i1, j2]
            I += [a, b, a, c]
            J += [b, a, c, a]
            idx2pos[pos2idx[i1, j1], :] = [i1, j1]
    I = np.array(I)
    J = np.array(J)
    V = np.ones(len(I))
    A = sparse.coo_matrix((V, (I, J)), shape=(N*N, N*N)).tocsr()
    return (A, pos2idx, idx2pos)


def draw2DGraph(A, idx2pos, vals, drawText = False):
    """
    Render the graph and its edges
    :param A: A sparse adjacency matrix
    :param idx2pos: An NVerticesx2 array of vertex positions
    :param vals: An NVertices-length array of scalar function values\
        assumed to be between 0 and 1
    :param drawText: If true, label the vertices by their index
    """
    c = plt.get_cmap('gray')
    C = c(np.array(np.round(vals*255.0), dtype=np.int32))
    C = C[:, 0:3]
    #First draw vertices
    plt.scatter(idx2pos[:, 0], idx2pos[:, 1], 200, c=C)
    if drawText:
        for i in range(A.shape[0]):
            pos = idx2pos[i, :]
            plt.text(pos[0]+0.15, pos[1]+0.15, "%i"%i)
    #Now draw edges
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
        X[i, :] = V1*(1-t) + V2*t
    return X

def getTauInterpolatedWindow(X, i1, Tau, dim):
    """
    Creates a window with the the Tau interpolated graphs given a starting 
        time index, Tau, and dimension, using linear interpolation
    :param X: An Nxd array of values at the scalar function 
    :param i1: Index in X where the window starts
    :param Tau: Length between frames in sliding window
    :param dim: Number of frames in sliding window
    :returns: XInterp: A dim x d sliding window array
    """
    i2 = int(np.ceil(i1 + Tau*dim))
    thisX = X[i1:i2+1, :]
    N = thisX.shape[0] #Number of frames in range of the window in the original video
    P = thisX.shape[1] #Number of nodes
    pix = np.arange(P)
    idx = np.arange(N)
    idxx = Tau*np.arange(dim)
    f = scipy.interpolate.interp2d(pix, idx, thisX, kind='linear')
    return f(pix, idxx)

def getRandomPosIdxs(K, pos2idx):
    """
    Choose a random 2x2 location on a KxK torus
    """
    #Pick a random start position for the square
    pos = np.random.random_integers(0, K-1, 2)
    pos = [ [pos[0], pos[1]], \
                     [pos[0]+1, pos[1]],\
                     [pos[0]+1, pos[1]+1],\
                     [pos[0], pos[1]+1] ]
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
    :returns {'XRet': A (TransLen*9*NPeriods) x (KxK) array of scalar values\
                    at the vertices of the torus at each point in time\
              'idx2pos':idx2pos, 'A':Adjacency matrix}
    """
    XRet = np.array([])
    (A, pos2idx, idx2pos) = getTorusAdj(K)
    N = A.shape[0]
    idxs = getRandomPosIdxs(K, pos2idx)
    for T in range(NPeriods):
        #First count down
        k = len(idxs)
        while k > 0:
            V1 = np.zeros(N)
            V1[idxs[0:k]] = 1.0
            V2 = np.zeros(N)
            V2[idxs[0:k-1]] = 1.0
            thisX = interpolateStates(V1, V2, TransLen)
            if XRet.size == 0:
                XRet = thisX
            else:
                XRet = np.concatenate((XRet, thisX), 0)
            k -= 1
        #Now count up at a different place
        idxs = getRandomPosIdxs(K, pos2idx)
        while k < len(idxs):
            V1 = np.zeros(N)
            V1[idxs[0:k]] = 1.0
            V2 = np.zeros(N)
            V2[idxs[0:k+1]] = 1.0
            thisX = interpolateStates(V1, V2, TransLen)
            XRet = np.concatenate((XRet, thisX), 0)
            k += 1
    return {'XRet':XRet, 'A':A, 'idx2pos':idx2pos, 'pos2idx':pos2idx}

if __name__ == '__main__':
    np.random.seed(10)
    K = 8
    NPeriods = 10
    TransLen = 5
    res = getBlinkingVideo(K, NPeriods, TransLen)
    [XRet, A, idx2pos] = [res['XRet'], res['A'], res['idx2pos']]
    
    draw2DGraph(A, idx2pos, np.zeros(A.shape[0]), True)
    plt.savefig("Indices.svg", bbox_inches='tight')
    
    #First plot that scalar function over time at each individual node
    plt.figure(figsize=(10, XRet.shape[1]))
    for j in range(XRet.shape[1]):
        plt.subplot(XRet.shape[1], 1, j+1)
        plt.plot(XRet[:, j])
        plt.ylim([-0.2, 1.2])
        if j == XRet.shape[1]-1:
            ax = plt.gca()
            ax.set_yticks([])
        else:
            plt.axis('off')
        plt.title("%i"%j)
    plt.tight_layout()
    plt.savefig("Plots.svg", bbox_inches='tight')
    
    plt.figure(figsize=(5, 5))
    #Now output video
    for i in range(XRet.shape[0]):
        plt.clf()
        draw2DGraph(A, idx2pos, XRet[i, :])
        plt.savefig("%i.png"%i, bbox_inches='tight')
