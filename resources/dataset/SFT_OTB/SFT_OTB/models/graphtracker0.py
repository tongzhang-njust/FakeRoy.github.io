'''
graphtracker.py
'''

import tensorflow as tf
import numpy as np
import scipy.sparse
from . import pyutis

class GTBase(object):
    def __init__(self):
        self.class_name = 'GTBase'

    def partition_input(self, L, x, m, p, d2, K):
        ## x: m*d
        ## o: List of length=p with m*(d2*K)
        xList = tf.split(1,p,x) # len=p list of m*d2
        ##
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat(0,[x,x_])

        oList = []
        for z0 in xList: # z0:m*d2
            #print('z0.shape={}'.format(z0.get_shape()))
            z  = tf.expand_dims(z0,0)  # 1*m*d2           
            if K > 1:
                #z1 = tf.sparse_tensor_dense_matmul(L, z0)
                z1 = tf.matmul(L, z0)
                z  = concat(z, z1)
            for k in range(2,K):
                z2 = 2*tf.matmul(L, z1) - z0
                #z2 = 2 * tf.sparse_tensor_dense_matmul(L, z1) - z0
                z  = concat(z, z2)
                z0, z1 = z1, z2

            ## z: K*m*d2
            z = tf.transpose(z, perm = [1,2,0])  # m*d2*K
            z = tf.reshape(z,[m,d2*K])  # m*(d2*K) 
            oList.append(z)
        return oList

class GTTr(GTBase):

    def __init__(self, m, d, K, p, gamma):
    # L: laplacian graph: m*m
    # m: num of nodes
    # d: fea dim
    # K: order of graph
    # p: num of partitioned experts
    # gamma: balance factor

        assert(d % p == 0)
        d2 = np.int32(d/p)

        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                #print('batch_size:{}, {}'.format(self.batch_size, M_0))
                self.ph_data = tf.placeholder(tf.float32,(m, d ),'data')
                self.ph_labels = tf.placeholder(tf.float32,(m,1),'labels')
                self.L   = tf.placeholder(tf.float32,(m,m),'L')


            #### gamma*I
            I_1 = np.eye(d2*K, dtype = np.float32)
            I_1 = tf.constant(I_1)
            self.gI = I_1*np.float32(gamma)
        
            #### L rescale
            I_2 = np.eye(m, dtype = np.float32)
            I_2 = tf.constant(I_2)
            lmax = 2
            ff = np.float32(2/lmax)
            self.L2 = self.L*ff - I_2

            #### convert x: m*d --> p[m*(d2*K)]
            self.xlist = self.partition_input(self.L2, self.ph_data, m, p, d2, K)
            #### solver
            ws = [] # p*(d2*K)*1
            for x in self.xlist: # x: m*(d2*K)
                #print('x.shape={}'.format(x.get_shape()))
                xtx = tf.matmul(x,x, transpose_a = True, transpose_b = False)
                xtx = xtx + self.gI
                invx = tf.matrix_inverse(xtx)
                xty = tf.matmul(x, self.ph_labels, transpose_a = True, transpose_b = False)
                ws.append( tf.expand_dims(tf.matmul(invx,xty),0))
            self.ws = tf.concat(0,ws)

        ##
        self.graph.finalize()

#######################################################

class GTTe(GTBase):

    def __init__(self, m, d, K, p):
    # L: laplacian graph: m*m
    # m: num of nodes
    # d: fea dim
    # K: order of graph
    # p: num of partitioned experts
    # gamma: balance factor

        assert(d % p == 0)
        d2 = np.int32(d/p)

        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                #print('batch_size:{}, {}'.format(self.batch_size, M_0))
                self.ph_data = tf.placeholder(tf.float32,(m, d),'data')
                self.ws = tf.placeholder(tf.float32,(p,d2*K,1),'ws')
                self.L   = tf.placeholder(tf.float32,(m,m),'L')

            #### L rescale
            I_2 = np.eye(m, dtype = np.float32)
            I_2 = tf.constant(I_2)
            lmax = 2
            ff = np.float32(2/lmax)
            self.L2 = self.L*ff - I_2

            #### convert x: m*d --> p[m*(d2*K)]
            self.xlist = self.partition_input(self.L2, self.ph_data, m, p, d2, K)
            #### ws
            #print('{}'.format(self.ws.get_shape()))
            self.ws2 = tf.split(0,p,self.ws)
            #### solver
            self.pred = [] # p[(d2*K)*1]
            for x,w in zip(self.xlist,self.ws2): # x: m*(d2*K)
                #print('{}:{}'.format(x.get_shape(),w.get_shape()))
                w = tf.reshape(w,[d2*K,1])
                y = tf.matmul(x,w)
                self.pred.append(tf.expand_dims(y,0))
            self.pred = tf.concat(0,self.pred)
        ##
        self.graph.finalize()

