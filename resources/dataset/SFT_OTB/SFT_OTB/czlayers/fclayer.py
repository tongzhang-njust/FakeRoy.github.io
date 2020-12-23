'''
fc.py
'''

import theano
import theano.tensor as T
import numpy as np
from utis import build_w_b

class FCLayer(object):
    #### input_shape: (batch_size,feadim)
    #### out_dim: (out_dim)
    def __init__(self,rng,input,layerparams):#input_shape,out_dim):
        self.input = input
	in_shape = layerparams['in_shape']
	out_dim = layerparams['out_dim']
        ##
	#W,b = build_w_b(rng,in_shape[1],out_dim,'gaussian',(0.0,0.01))
        W,b = build_w_b(rng,in_shape[1],out_dim,'uniform',1.)	
	self.W = theano.shared(W,borrow=True)
	self.b = theano.shared(b,borrow = True)

	RELU = lambda x: x*(x>0)
	self.output = T.tanh(T.dot(input,self.W)+self.b.dimshuffle('x',0))
	self.params = [self.W, self.b]
	##
	self.gW = theano.shared(np.zeros_like(W),borrow = True)
        self.gb = theano.shared(np.zeros_like(b),borrow = True)
        self.gparams = [self.gW,self.gb]

