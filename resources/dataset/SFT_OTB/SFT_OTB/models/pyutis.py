'''
pyutis.py
'''

import numpy as np
import scipy.sparse

'''
rescale_L(L,lmax=2)
'''

def zeros_sparse_matrix(h,w):

    return scipy.sparse.csr_matrix((h,w),dtype=np.float32)

def rescale_L(L,lmax=2):
    M,M = L.shape
    I = scipy.sparse.identity(M, format='csr',dtype=L.dtype)
    L /= lmax/2
    L -= I
    return L
