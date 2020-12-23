'''
graphtracker.py
'''

import tensorflow as tf
import numpy as np
import scipy.sparse
from . import pyutis

class GTBase(object):
    def __init__(self,L, m):

        assert(L.shape[0] == m and L.shape[1] == m)

        L = scipy.sparse.csr_matrix(L)
        L = pyutis.rescale_L(L,lmax=2)
        self.L = L.tocoo()
        self.indices = np.column_stack((self.L.row,self.L.col))
        

    def partition_input(self, L, x, m, map_idx, K):
        ## x: m*d
        ## o: List of length=p with m*(d2*K)
        self.xList = tf.split(x,map_idx[1:]-map_idx[:-1],axis=1)
        ##
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat([x,x_],0)

        oList = []
        for z0 in self.xList: # z0:m*d2
            z  = tf.expand_dims(z0,0)  # 1*m*d2           

            if K > 1:
                z1 = tf.sparse_tensor_dense_matmul(L, z0)
                z  = concat(z, z1)
            for k in range(2,K):
                z2 = 2 * tf.sparse_tensor_dense_matmul(L, z1) - z0
                z  = concat(z, z2)
                z0, z1 = z1, z2

            ## z: K*m*d2
            z = tf.transpose(z, perm = [1,2,0])  # m*d2*K
            z = tf.reshape(z,[m,-1])  # m*(d2*K) 

            oList.append(z)
        return oList

class GTTr(GTBase):

    def __init__(self, L, m, map_idx, K, gamma,fast_max_iter = 5):
    # L: laplacian graph: m*m
    # m: num of nodes
    # map_idx: indices of maps representing feature dims
    # K: order of graph
    # p: num of partitioned experts
    # gamma: balance factor

        #super().__init__(L, m)
        GTBase.__init__(self,L, m)
        self.p = len(map_idx)-1
        self.d = map_idx[-1]

        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                self.ph_data = tf.placeholder(tf.float32,(m, self.d ),'data')
                self.ph_labels = tf.placeholder(tf.float32,(m,1),'labels')
                self.ph_wt = tf.placeholder(tf.float32,(K*self.d,1),'wt')  # 

            self.L2 = tf.SparseTensor(self.indices, self.L.data, self.L.shape)
            self.L2 = tf.sparse_reorder(self.L2)
            self.gamma = tf.constant(np.float32(gamma))

            #### convert x: m*d --> p[m*(d2*K)]
            self.xlist = self.partition_input(self.L2, self.ph_data, m, map_idx, K)

            #### solver
            ws = [] # (p*(d2*K))*1
            for x in self.xlist: # x: (m*(d2*K)
                xtx = tf.matmul(x,x, transpose_a = True, transpose_b = False)
                xtx_diag = tf.matrix_diag_part(xtx) + self.gamma#/num_map
                xtx = tf.matrix_set_diag(xtx,xtx_diag)
                invx = tf.matrix_inverse(xtx)
                xty = tf.matmul(x, self.ph_labels, transpose_a = True, transpose_b = False)
                ws.append(tf.matmul(invx,xty))
            self.ws = tf.concat(ws,0)

            #### fast
            ws_fast = [] # (p*(d2*K))*1
            ws_old = tf.split(self.ph_wt,K*map_idx[1:]-K*map_idx[:-1],axis=0) 
            for x,w_old in zip(self.xlist,ws_old):  
                xtx = tf.matmul(x,x,transpose_a=True,transpose_b=False)
                xty = tf.matmul(x, self.ph_labels, transpose_a = True, transpose_b = False)
                theta = w_old 
                for k in range(fast_max_iter):
                    M = tf.matmul(xtx,theta) - xty + (2*self.gamma)*theta
                    den1 = tf.matmul(theta,xtx,transpose_a = True, transpose_b = False)
                    den  = tf.matmul(den1,M)-tf.matmul(xty,M, transpose_a= True, transpose_b=False)\
					+ 2*self.gamma*tf.matmul(theta,M,transpose_a = True, transpose_b = False)
                    num1 = tf.matmul(M,xtx,transpose_a = True, transpose_b = False)
                    num  = tf.matmul(num1,M) + 2*self.gamma*tf.matmul(M,M,transpose_a = True, transpose_b = False)
                    eta  = tf.div(den,num)
                    theta = theta - eta*M
                ws_fast.append(theta)
            self.ws_fast = tf.concat(ws_fast,0)

        self.graph.finalize()

#######################################################

class GTTe(GTBase):

    def __init__(self,L, m, map_idx, K):
    # L: laplacian graph: m*m
    # m: num of nodes
    # d: fea dim
    # K: order of graph
    # p: num of partitioned experts
    # gamma: balance factor

        #super().__init__(L, m)
        GTBase.__init__(self,L, m)
        self.p = len(map_idx)-1
        self.d = map_idx[-1]

        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                self.ph_data = tf.placeholder(tf.float32,(m, self.d),'data')
                self.ws = tf.placeholder(tf.float32,(self.d*K,1),'ws')
            
            self.L2 = tf.SparseTensor(self.indices, self.L.data, self.L.shape)
            self.L2 = tf.sparse_reorder(self.L2)

            #### convert x: m*d --> p[m*(d2*K)]
            self.xlist = self.partition_input(self.L2, self.ph_data, m, map_idx, K)
            #### ws
            self.ws2 = tf.split(self.ws,K*map_idx[1:]-K*map_idx[:-1],axis=0)
            #### solver
            self.pred = [] # p[m*1]
            for x,w in zip(self.xlist,self.ws2): # x: m*(d2*K)
                y = tf.matmul(x,w)
                self.pred.append(tf.expand_dims(y,0))
            self.pred = tf.concat(self.pred,0)
        ##
        self.graph.finalize()


class GTGK(GTBase):

    def __init__(self,gamma, L, m, d, K, p):
    # L: laplacian graph: m*m
    # m: num of nodes
    # d: fea dim
    # K: order of graph
    # p: num of partitioned experts
    # gamma: balance factor

        super().__init__(L, m, d, p)

        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                #print('batch_size:{}, {}'.format(self.batch_size, M_0))
                self.ph_data = tf.placeholder(tf.float32,(m, d ),'data')
                #self.ph_labels = tf.placeholder(tf.float32,(m, 1),'labels')
                #self.L   = tf.placeholder(tf.float32,(m,m),'L')


            self.L2 = tf.SparseTensor(self.indices, self.L.data, self.L.shape)
            self.L2 = tf.sparse_reorder(self.L2)

            #### gamma*I
            I_1 = np.eye(self.d2*K, dtype = np.float32)
            I_1 = tf.constant(I_1)
            self.gI = I_1*np.float32(gamma)

            #### convert x: m*d --> p[m*(d2*K)]
            self.xlist = self.partition_input(self.L2, self.ph_data, m, p, self.d2, K)
            #for x in self.xlist: # x: m*(d2*K)
            #    xtx = tf.matmul(x,x, transpose_a = True, transpose_b = False)
            #    xtx = xtx + self.gI
            #    invx = tf.matrix_inverse(xtx)
            #    xty = tf.matmul(x, self.ph_labels, transpose_a = True, transpose_b = False)
            #    ws.append( tf.expand_dims(tf.matmul(invx,xty),0))
            #self.ws = tf.concat(0,ws)
            #self.x = tf.stack(self.xlist,axis=0) # nexperts*m*d2

        self.graph.finalize()

