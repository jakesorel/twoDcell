import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _tnorm(x):
    return np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)


@jit(nopython=True, cache=True)
def _roll(x, direc=1):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)

    :param x:
    :return:
    """
    if direc == -1:
        return np.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1:
        return np.column_stack((x[:, 1:3], x[:, 0]))

@jit(nopython=True)
def _roll3(x,direc=1):
    x_out = np.empty_like(x)
    x_out[:,:,0],x_out[:,:,1] = _roll(x[:,:,0],direc=direc),_roll(x[:,:,1],direc=direc)
    return x_out

def _CV_matrix(tri_list, n_v, n_c):
    CV_matrix = np.zeros((n_c, n_v, 3))
    for i in range(3):
        CV_matrix[tri_list[:, i], np.arange(n_v), i] = 1
    return CV_matrix

@jit(nopython=True, cache=True)
def _tri_sum(n_c,CV_matrix,tval):
    val_sum = np.zeros(n_c)
    for i in range(3):
        val_sum += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(tval[:,i])
    return val_sum

@jit(nopython=True)
def _cosine_rule(a,b,c):
    return np.arccos((b**2 + c**2 - a**2)/(2*b*c))

@jit(nopython=True)
def _clip(x,xmin,xmax):
    xflat = x.ravel()
    minmask = xflat<xmin
    maxmask = xflat>xmax
    xflat[minmask] = xmin
    xflat[maxmask] = xmax
    return xflat.reshape(x.shape)

@jit(nopython=True)
def _replace_val(x,mask,xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew
    return xflat.reshape(x.shape)

@jit(nopython=True)
def _replace_vec(x,mask,xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew.ravel()[maskflat]
    return xflat.reshape(x.shape)

@jit(nopython=True)
def _tcross(A,B):
    return A[:,:,0]*B[:,:,1] - A[:,:,1]*B[:,:,0]

@jit(nopython=True)
def _tdot(A,B):
    return A[:,:,0]*B[:,:,0] + A[:,:,1]*B[:,:,1]

@jit(nopython=True)
def _touter(A,B):
    return np.dstack((np.dstack((A[:,:,0]*B[:,:,0],A[:,:,1]*B[:,:,0])),
                      np.dstack((A[:,:,0]*B[:,:,1],A[:,:,1]*B[:,:,1])))).reshape(-1,3,2,2)

@jit(nopython=True)
def _tidentity(nv):
    I = np.zeros((nv,3,2,2))
    I[:,:,0,0] = 1
    I[:,:,1,1] = 1
    return I

@jit(nopython=True)
def _tmatmul(A,B):
    """

    Check this...

    :param A:
    :param B:
    :return:
    """
    AT,BT = A.T,B.T
    return np.dstack(((AT[0]*BT[0,0]+AT[1]*BT[1,0]).T,
                      (AT[0]*BT[0,1]+AT[1]*BT[1,1]).T))

@jit(nopython=True)
def _find_neighbour_val(A,neighbours):
    """
    Check this
    :param A:
    :param neighbours:
    :return:
    """
    B = np.empty_like(A)
    for i, tneighbour in enumerate(neighbours):
        for j, neighbour in enumerate(tneighbour):
            B[i,j] = A[neighbour,j]
    return B

@jit(nopython=True)
def _repeat_mat(A):
    return np.dstack((A,A,A,A)).reshape(-1,3,2,2)

@jit(nopython=True)
def _repeat_vec(A):
    return np.dstack((A,A))


#
# @jit(nopython=True)
# def _get_edge_list(tri_list):
#     edges = np.concatenate((tri_list[:,:2],tri_list[:,1:],np.column_stack((tri_list[:,0],tri_list[:,1]))),axis=0)
#     return edges
#
# def _get_b_edges(tri_list):
#     edges = _get_edge_list(tri_list)
#     edges.sort(axis=1)
#     sorted_idx = np.lexsort(edges.T)
#     sorted_data = edges[sorted_idx,:]
#     compare = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
#     row_mask = np.roll(compare,-1)*compare
#     return sorted_data[row_mask]

@jit(nopython=True)
def _get_b_edges(CV_matrix,n_c):
    CCW_matrix = np.zeros((n_c,n_c))
    for i in range(3):
        CCW_matrix += np.asfortranarray(CV_matrix[:,:,i])@np.asfortranarray(CV_matrix[:,:,np.mod(i+1,3)].T)
    return np.nonzero((CCW_matrix-CCW_matrix.T) ==-1) #absent in CCW matrix, present in the CW matrix


def normalise(x):
    return (x-x.min())/(x.max()-x.min())

def make_circular_boundary(S,N, dist):
    centroid = S.mean(axis=0)
    mdist = np.linalg.norm(S - centroid,axis=1).max()
    r = dist+mdist
    phi = np.arange(0,2*np.pi,2*np.pi/N)
    S_bound = np.array([centroid[0]+r*np.cos(phi),centroid[1]+r*np.sin(phi)]).T
    return S_bound

def hexagonal_lattice(rows=3, cols=3, noise=0.05):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    points = []
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    return points

def square_boundary(xlim,ylim):
    return np.array([[xlim[0],ylim[0]],[xlim[0],ylim[1]],[xlim[1],ylim[0]],[xlim[1],ylim[1]]])
#
# @jit(nopython=True,cache=True)
# def _get_neighbours(tri_list,neigh=None):
#     """
#     Given a triangulation, find the neighbouring triangles of each triangle.
#     By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.
#     Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)
#     :param tri: Triangulation (n_v x 3) np.int32 array
#     :param neigh: neighbourhood matrix to update {Optional}
#     :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
#     """
#     n_v = tri_list.shape[0]
#     if neigh is None:
#         neigh = np.ones_like(tri_list,dtype=np.int32)*-1
#     tri_compare = np.concatenate((tri_list.T, tri_list.T)).T.reshape((-1, 3, 2))
#     for j in range(n_v):
#         tri_sample_flip = np.flip(tri_list[j])
#         tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
#         for k in range(3):
#             if neigh[j,k]==-1:
#                 loc_mask = (tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1])
#                 if loc_mask.sum()!=0:
#                     neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
#                     neighb,l = neighb[0],l[0]
#                     neigh[j,k] = neighb
#                     neigh[neighb,np.mod(2-l,3)] = j
#     return neigh
#


@jit(nopython=True,cache=True)
def _get_neighbours(tri_list,neigh=None,ls = None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.
    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.
    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri_list.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri_list,dtype=np.int32)*-1
    if ls is None:
        ls = np.ones_like(tri_list,dtype=np.int32)*-1
    tri_compare = np.concatenate((tri_list.T, tri_list.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri_list[j])
        tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                loc_mask = (tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1])
                if loc_mask.sum()!=0:
                    neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
                    neighb,l = neighb[0],l[0]
                    neigh[j,k] = neighb
                    ls[j,k] = l
                    # neigh[neighb,np.mod(2-l,3)] = j
                    # ls[neighb, np.mod(2 - l, 3)] = k
    return neigh,ls


def _cell_only_tri_list(tri_list,n_b,n_c):
    return tri_list[(tri_list<n_c - n_b).all(axis=1)]