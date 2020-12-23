'''
cgcnn.py
'''

import tensorflow as tf
import numpy as np
import scipy.sparse
import os
from . import pyutis#, base_model #import * #rescale_L
from .base_model import BaseModel
#import base_model

class cgcnn(BaseModel):
    
    def __init__(self, num_node, Fin, F, K, p = None, M = None, filter = 'chebyshev5', brelu = None, pool = None, \
                 num_epochs = 20, learning_rate = 0.1, decay_rate = 0.95, decay_step = None, momentum = 0.9,\
                 regularization = 0, dropout = 0, batch_size = 128, eval_frequency = 200, dir_name = ''):
        
        super().__init__()     
   
        ## verfiy the consistency
        assert len(F) == len(K) >= len(p)

        self.nnode_layers = [num_node]

        ## print info about NN
        n_gcnn = len(p)
        n_fc   = len(M)
        M_0 = num_node
        print('NN architecture...')
        print('	 input: M_0 = {}'.format(M_0))
        self.L = []
        for ii in range(n_gcnn):
            print('  *layer {0}: cgconv{0}'.format(ii))
            M_1 = M_0//p[ii]
            print('    L_{0}.shape[0]={6}; reprentation: M_{0}*F_{1}/p_{1} = {2}*{3}/{4} = {5}'.format(ii,ii, M_0, F[ii], p[ii], \
                                                        M_0*F[ii]//p[ii],M_1))

            self.L.append(scipy.sparse.csr_matrix((M_0,M_0),dtype=np.float32))
            self.nnode_layers.append(M_1)
            M_0 = M_1

            F_last = F[ii-1] if ii > 0 else Fin
            print('    weights: F_{0}*F{1}*K_{1} = {2}*{3}*{4}={5}'.format(ii, ii, F_last, F[ii], K[ii], F_last*F[ii]*K[ii]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(ii, F[ii]))
            else:
                assert brelu == None or brelu == 'b1relu'
        self.L.append(scipy.sparse.csr_matrix((M_0,M_0),dtype=np.float32))

        for ii in range(n_fc):
            name = 'logits (softmax)' if ii == n_fc-1 else 'fc{}'.format(ii)
            print('  layer {}: {}'.format(n_gcnn+ii, name))
            print('    representation: M_{} = {}'.format(n_gcnn+ii, M[ii]))
            M_last = M[ii-1] if ii>0 else M_0 if n_gcnn == 0 else L[-1].shape[0]*F[-1]//p[-1]
            print('    weights: M_{}*M_{} = {}*{}={}'.format(n_gcnn + ii, n_gcnn+1, M_last, M[ii], M_last*M[ii] ))
            print('    biases: M_{} = {}'.format(n_gcnn+ii, M[ii]))

        ## strore attributes
        self.Fin, self.F, self.K, self.p, self.M = Fin, F,K,p,M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_step, self.momentum = decay_rate, decay_step, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.is_brelu = True
        if brelu == 'b1relu':
            self.brelu = getattr(self, brelu)
        else:
            assert brelu == None
            self.is_brelu = False

        self.is_pool = True
        if len(p) > 1 or p[0] > 1:
            self.pool  = getattr(self, pool)
        else:
            self.is_pool = False
        
        ## build the computational graph
        self.build_graph(num_node, Fin)

    def set_L(self, L):
        self.L = []
        for Li in L:
            self.L.append(scipy.sparse.csr_matrix(Li))

    def build_graph(self, M_0, d):
        "build the computational graph of the model"
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            ## input
            with tf.name_scope('input'):
                #print('batch_size:{}, {}'.format(self.batch_size, M_0))
                self.ph_data = tf.placeholder(tf.float32,(self.batch_size,M_0, d ),'data')
                self.ph_labels = tf.placeholder(tf.float32,(self.batch_size,M_0),'labels')
                self.ph_dropout = tf.placeholder(tf.float32,(),'dropout')

            ## model -- forward inference
            op_logits = self.inference(self.ph_data, self.ph_dropout) # N*(M*d)
            self.op_loss, self.op_loss_average = self._euclidean_loss(op_logits, self.ph_labels, self.regularization)
            #self.op_loss, self.op_loss_average = self._cross_entropy_loss(op_logits, self.ph_labels, self.regularization)

            ## training
            self.op_train = self.training(self.op_loss, self.learning_rate, self.decay_step, self.decay_rate, self.momentum)             

            ## prection 
            self.op_prediction = op_logits
            #self.op_prediction = self.prediction(op_logits)

            ## intialize variables
            self.op_init = tf.initialize_all_variables()

            ## summaries
            self.op_summary = tf.merge_all_summaries()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def training(self,loss, learning_rate, decay_steps, decay_rate =0.95, momentum = 0.9):

        with tf.name_scope('training'): 
            ## learning rate
            global_step = tf.Variable(0, name='global_step', trainable = False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase = True)
            tf.scalar_summary('learning_rate', learning_rate)
            
            ## optimizer
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step = global_step)

            ## Histograms
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.histogram_summary(var.op.name+'/gradient', grad)

            ## 
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
          
        return op_train
 
                    
    def prediction(self, logits):
        
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, dimension = 1)

            return prediction
            

    def inference(self, x, dropout):
        ## x: n*m*f
        
        ## graph covn layers
        #x = tf.expand_dims(x,2) # n*m*(f=1)

        for ii in range(len(self.p)):
            with tf.variable_scope('conv_{}'.format(ii)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[ii], self.F[ii], self.K[ii])
                if self.is_brelu:
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                if self.is_pool:
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.p[ii]) 

        ## fc layers
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])
        for ii, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc_{}'.format(ii+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        ## logits linear layer
        if len(self.M) > 0:
            with tf.variable_scope('logits'):
                x = self.fc(x, self.M[-1], relu = False)

        return x # N*(M*C=1/d)

    
    def fc(self, x, Mout, relu = True):

        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization = True)
        b = self._bias_variable([1,Mout], regularization = False) # ?? False
        x = tf.matmul(x, W) + b 

        return tf.nn.relu(x) if relu else x

    def chebyshev5(self, x, L, Fout, K):
        # L: n*m*m 
        # x: n*m*d

        N,M,Fin = x.get_shape()
        N,M,Fin = int(N),int(M),int(Fin)

        ## rescale L and store as a TF sparse tensor
        L = scipy.sparse.csr_matrix(L)
        L = pyutis.rescale_L(L,lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row,L.col))

        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        ## Transform to Chebyshev basis
        x0 = tf.transpose(x,perm=[1,2,0]) # M*Fin*N
        x0 = tf.reshape(x0,[M, Fin*N]) # M*(Fin*N)
        x  = tf.expand_dims(x0,0)  # 1*M*(Fin*N)

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat(0,[x,x_])

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x  = concat(x, x1)
        for k in range(2,K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
            x  = concat(x, x2)
            x0, x1 = x1, x2
        # x: K*M*(Fin*N)
        x = tf.reshape(x,[K, M, Fin, N])
        x = tf.transpose(x, perm = [3,1,2,0])
        x = tf.reshape(x, [N*M, Fin*K])
       
        ## Filter
        W = self._weight_variable([Fin*K, Fout], regularization = True)
        x = tf.matmul(x, W)

        return tf.reshape(x, [N, M, Fout])


    def b1relu(self, x):
        N, M, F = x.get_shape()
        b = self._bias_variable([1,1,int(F)], regularization = False)

        return tf.nn.relu(x + b)


    def mpool1(self, x, p):
        
        if p > 1:
            x = tf.expand_dims(x,3) # N*M*F*1
            x = tf.nn.max_pool(x, ksize = [1,p,1,1], strides = [1,p,1,1],padding = 'SAME')
            return tf.squeeze(x,[3]) 
        else:
            return x


        

        




