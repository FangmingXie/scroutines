import leidenalg as la
import numpy as np
import pynndescent
from scipy import sparse
from scipy.cluster import hierarchy as sch

def adjacency_to_igraph(adj_mtx, weighted=False, directed=True, simplify=True):
    """
    Converts an adjacency matrix to an igraph object
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        directed (bool): If graph should be directed
    
    Returns:
        G (igraph object): igraph object of adjacency matrix
    
    Uses code from:
        https://github.com/igraph/python-igraph/issues/168
        https://stackoverflow.com/questions/29655111

    Author:
        Wayne Doyle 
        (Fangming Xie modified) 
    """
    nrow, ncol = adj_mtx.shape
    if nrow != ncol:
        raise ValueError('Adjacency matrix should be a square matrix')
    vcount = nrow
    sources, targets = adj_mtx.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = igraph.Graph(n=vcount, edges=edgelist, directed=directed)
    if weighted:
        G.es['weight'] = adj_mtx.data
    if simplify:
        G.simplify() # simplify inplace; remove duplicated and self connection (Important to avoid double counting from adj_mtx)
    return G

def leiden(G, cells,
           resolution=1, seed=0, n_iteration=2,
           **kwargs,
          ):
    """cells are in order of the Graph node order -- whatever it is
    """
    partition = la.find_partition(G, 
                                  la.RBConfigurationVertexPartition, # modularity with resolution
                                  resolution_parameter=resolution, seed=seed, n_iterations=n_iteration, **kwargs)
    # get cluster labels from partition
    labels = [0]*(len(cells)) 
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i+1
    return labels

def is_in_polygon(poly, ps):
    """
    test if each point is in the defined polygon
    poly: array-like (m,2)
    ps: array-like (n,2)
    """
    # is_in = False
    poly = np.asarray(poly) 
    npoly = len(poly)
    
    ps = np.asarray(ps)
    is_in = np.zeros(len(ps), dtype=int)
    
    for i in range(npoly):
        j = (i-1) % npoly
        if poly[j,1]-poly[i,1] != 0: # not cross
            cond1 = ((poly[i,1]>ps[:,1]) != (poly[j,1]>ps[:,1]))
            cond2 = (ps[:,0] < poly[i,0] + (ps[:,1]-poly[i,1])*(poly[j,0]-poly[i,0])/(poly[j,1]-poly[i,1]))
            cond = np.logical_and(cond1, cond2)
            is_in += cond  
    return (is_in % 2).astype(bool)

def build_feature_graph_knnlite(ftrs_mat, k=15, metric='cosine'):
    """
    """
    N = len(ftrs_mat)
    
    # kNN graph
    knn = pynndescent.NNDescent(ftrs_mat,
                                n_neighbors=k,
                                metric=metric,
                                diversify_prob=1,
                                pruning_degree_multiplier=1.5,
                                )
    idx, _ = knn.neighbor_graph

    # to adj and to graph
    i = np.repeat(np.arange(N), k-1)
    j = idx[:,1:].reshape(-1,)
    adj_mat = sparse.coo_matrix((np.repeat(1, len(i)), (i,j)), shape=(N,N))
    G = adjacency_to_igraph(adj_mat, directed=False, simplify=True)
    
    return G
