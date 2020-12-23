'''
coarsen.py
'''

import numpy as np
import scipy.sparse

'''
coarsen(A,levels,self_connections=False)
'''

def perm_data(x,indices):
    if indices is None:
        return x

    n, d = x.shape
    dnew = len(indices)
    assert dnew >= d
    xnew = np.empty((n,dnew))

    for ii,jj in enumerate(indices):
        if jj < d:
            xnew[:,ii] = x[:,jj]
        else:
            xnew[:,ii] = np.zeros(n)
    return xnew

def coarsen(A,levels,self_connections=False):

    # A: graph metrix with no isolated points    

    graphs,parents = metis(A,levels)

    perms = compute_perm(parents)

    ##
    for ii, A in enumerate(graphs):
        M, M = A.shape
        
        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if ii < levels:
            A = perm_adjacency(A, perms[ii])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[ii] = A
        
        Mnew, Mnew = A.shape
        print('layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
		'|E| = {3} edges'.format(ii, Mnew, Mnew-M, A.nnz//2))
       
    return graphs, perms[0] if levels > 0 else None
        


def perm_adjacency(A,indices):

    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M, M), dtype = np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype = np.float32)
        A = scipy.sparse.vstack([A,rows])
        A = scipy.sparse.hstack([A,cols])


    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col] # val

    assert type(A) is scipy.sparse.coo.coo_matrix
    return A

    
def compute_perm(parents):
    
    indices = []
    if len(parents) > 0:
        M_last = np.max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        
        pool_singeltons = len(parent)
        indices_layer = []
        for ii in indices[-1]:
            indices_node = list(np.where(parent == ii)[0])
            assert 0 <= len(indices_node) <= 2

            if(len(indices_node)) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
            elif len(indices_node) == 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
            indices_layer.extend(indices_node)

        indices.append(indices_layer)

    ## sanity check
    for i, indices_layer in enumerate(indices):
        M = M_last*(2**i)
        assert len(indices[0] == M) # useless
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


def metis(W,levels,rid=None):
    # W: N*N
    # levels: number of graphs
    # randperm: disorder the graph nodes
    # return: graphs: list, parents: list 

    N,N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    graphs  = []
    graphs.append(W)

    degree = W.sum(axis=0) - W.diagonal()
    for _ in range(levels):
        weights = degree
        weights = np.array(weights).squeeze()
     
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)

        parents.append(cluster_id)

	## merge cluster nodes
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)),shape=(Nnew,Nnew)) 
        W.eliminate_zeros()

        graphs.append(W)
        
        N, N = W.shape
        degree = W.sum(axis=0) # - W.diagonal()

        # generate rid ??
        ss = np.array(W.sum(axis=0)).squeeze() # maybe not use squeeze when using at the beggining/the first time
        rid = np.argsort(ss)

    return graphs, parents




def metis_one_level(rr,cc,val,rid,weights):

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N,np.int32)
    cluster_id = np.zeros(N,np.int32)

    oldval = rr[0]
    count = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            count = count + 1
            oldval = rr[ii]
            rowstart[count] = ii


    marked = np.zeros(N,np.bool)
    cluster_id = np.zeros(N,np.int32)
    clustercount = 0
    for ii in range(N):
        tid = rid[ii]

        if not marked[tid]:
            rs = rowstart[tid]
            wmax = 0.0
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = val[rs+jj]*(1.0/weights[tid]+1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            marked[tid] = True
            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True
            clustercount += 1

    return cluster_id











	


 
