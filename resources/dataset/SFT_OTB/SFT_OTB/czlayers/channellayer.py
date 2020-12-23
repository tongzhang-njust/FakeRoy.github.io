'''
channellayer.py
'''

import theano
import theano.tensor as T
import numpy as np
#from utis import build_w_b

class ChannelLayer(object):
    #### input_shape: (n,c,h,w)
    #### out_dim: (n,h*w,c)
    def __init__(self,rng,input,layerparams):#input_shape,out_dim):
        self.input = input
	in_shape = layerparams['in_shape']
	out_dim = layerparams['out_shape']
        ##
	self.output = input.dimshuffle((0,2,3,1))
	self.output = input.reshape(out_dim) #T.tanh(T.dot(input,self.W)+self.b.dimshuffle('x',0))
	self.params = []
	##
        self.gparams = []

