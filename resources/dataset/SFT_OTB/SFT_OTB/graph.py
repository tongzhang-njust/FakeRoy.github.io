'''
graph.py
'''

import sklearn.metrics
import sklearn.neighbors
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np
import pdb

'''
adjacency(dist,idx)
distance_sklearn_metrics(z,k=4,metric='euclidean')
extract_L(L, p)
fea_graph(x,num_edges = 6)
grid(m,dtype=np.float32)
grid_graph(params_gh,corners=False)
laplacian(W, normalized = True)
replace_random_edges(A,noise_level)
'''

def extract_L(L, p):
    assert L is list
    assert len(L) >= len(p)
    assert np.all(np.array(p) >= 1)
    p_log2 = np.where(np.array(p)>1, np.log2(p), 0)
    print(p_log2)
    assert np.all(np.mod(p_log2,1) == 0)
    assert len(L) >= 1 + np.sum(p_log2)

    j = 0
    L2 = []
    for pp in p:
        L2.append(L[j])
        j += int(np.log2(pp)) if pp > 1 else 0
    
    return L2

def fea_graph(x,num_edges = 6):
    # x: nsamp * nnode * d

    assert num_edges > 0
    ns, nn, d = x.shape
    if num_edges < 1:
        num_edges = np.max((1,np.floor(num_edges*nn)))
    num_edges = np.int32(num_edges)    

    A = np.zeros((ns,nn,nn),dtype = np.float32)
    for ii in range(ns):
        dist, idx = distance_sklearn_metrics(x[ii],k = num_edges,metric = 'euclidean')
        Ai = adjacency(dist,idx)
        A[ii,:,:] = Ai.todense()

    #A = scipy.sparse.
    return A    

def laplacian(W, normalized = True):
    d = W.sum(axis=0)

    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(),0) 
        L = D - W
    else:
        d += np.spacing(np.array(0,W.dtype))
        d  = 1/np.sqrt(d)
        D  = scipy.sparse.diags(d.A.squeeze(),0)
        I  = scipy.sparse.identity(d.size,dtype = W.dtype)
        L = I - D * W * D

    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def replace_random_edges(A,noise_level):
    M, M = A.shape
    n = int(noise_level*A.nnz//2)
    
    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0,M,n)
    cols = np.random.randint(0,M,n)
    vals = np.random.uniform(0,1,n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A,format='coo')
    assert A_coo.nnz == A.nnz//2
    assert A_coo.nnz >= n
 
    A = A.tolil()

    for idx, row, col, val in zip(indices,rows,cols,vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row,col] = val # ??
        A[col,row] = val # ??

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros() # eliminate isolated point
    return A
    

def grid_graph(params_gh,corners=False):
    ## params_gh: params of graph, dict, number_nodes number_edges, metric, 
    h,w  = params_gh['height_width']
    z = grid(h,w)
    if params_gh['number_edges'] < 1:
        K = np.max((1,np.floor(params_gh['number_edges']*h*w)))
    else:
        K = params_gh['number_edges']
    K = np.int32(K)

    dist,idx = distance_sklearn_metrics(z,K,params_gh['metric'])
    A = adjacency(dist,idx)

    ## corner vertices are connected to 2 neightbors only
    if corners:
        A = A.toarray()
        A[A<A.max()/1.5] = 0 # ??
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))	

    #print("{}".format(type(A)))
    print("{} > {} edges".format(A.nnz//2,params_gh['number_edges']*(h*w)))
    return A

def grid(h,w,dtype=np.float32):
    M = h*w
    x = np.linspace(0,1,h,dtype=dtype)
    y = np.linspace(0,1,w,dtype=dtype)
    xx,yy = np.meshgrid(x,y)
    z = np.empty((M,2),dtype)
    #pdb.set_trace()
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z

def distance_sklearn_metrics(z,k=4,metric='euclidean'):
    '''
    d = sklearn.metrics.pairwise.pairwise_distances(z,metric=metric, n_jobs=-2)
    #pdb.set_trace()
    min_d=float('%.5f'%(d[0,58]))
    max_d=float('%.5f'%(d[0,59]))
    for i in range((z.shape[0])):
        for j in range((z.shape[0])):
            if float('%.5f'%(d[i,j]))<=min_d or float('%.5f'%(d[i,j]))>max_d:
               d[i,j]=10
    #pdb.set_trace() 
    #idx = np.argsort(d)[:,1:k+1] 
    idx = np.argsort(d)[:,0:k] 
    d.sort()
    #d = d[:,1:k+1]
    d = d[:,0:k]
    return d,idx
    '''
    d = sklearn.metrics.pairwise.pairwise_distances(z,metric=metric, n_jobs=-2)
    #pdb.set_trace()
    min_d=float('%.2f'%d[0,58])#0.03
    max_d=float('%.2f'%d[0,59]) #0.04
    for i in range((z.shape[0])):
        for j in range((z.shape[0])):
            if (d[i,j])<=min_d or (d[i,j])>max_d:
               d[i,j]=10
    #pdb.set_trace() 
    #idx = np.argsort(d)[:,1:k+1] 
    idx = np.argsort(d)[:,0:k] 
    d.sort()
    #d = d[:,1:k+1]
    d = d[:,0:k]
    return d,idx   

def adjacency(dist,idx):
    M,k = dist.shape
    assert M,k == idx.shape
    assert dist.min() >= 0
    
    sigma2 = np.mean(dist[:,-1])**2
    dist   = np.exp(-dist**2/sigma2)

    I = np.arange(0,M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V,(I,J)),shape=(M,M)) 

    W.setdiag(0)

    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W-W.T).mean() < 1.0e-10
    assert type(W) is scipy.sparse.csr.csr_matrix

    return W
       





