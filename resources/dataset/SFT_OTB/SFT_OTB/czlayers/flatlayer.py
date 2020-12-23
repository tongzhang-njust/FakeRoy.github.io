'''
flatlayer.py
'''

import theano
import theano.tensor as T
import numpy as np
#from utis import build_w_b

class FlatLayer(object):
    #### input_shape: (batch_size,feadim)
    #### out_dim: (out_dim)
    def __init__(self,rng,input,layerparams):#input_shape,out_dim):
        self.input = input
	in_shape = layerparams['in_shape']
	out_dim = layerparams['out_shape']
        ##
	self.output = input.reshape(out_dim) #T.tanh(T.dot(input,self.W)+self.b.dimshuffle('x',0))
	self.params = []
	##
        self.gparams = []

