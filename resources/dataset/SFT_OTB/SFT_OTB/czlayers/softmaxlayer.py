'''
softmaxlayer.py
revised from logreg2layer and no learn projections
'''

import numpy as np
import theano
import theano.tensor as T
from utis import build_w_b

class SoftMaxLayer(object):

    def __init__(self, rng, input, layerparams):#input_shape, out_dim):
	in_shape = layerparams['in_shape'] # [n,nclass,h,w]
	#out_dim  = layerparams['class_num']
	#W,b = build_w_b(rng,in_shape[1],out_dim,'gaussian',(0.0,0.01))
        #W,b = build_w_b(rng,in_shape[1],out_dim,'uniform',1.)
        #self.W = theano.shared(W,borrow = True)
        #self.b = theano.shared(b,borrow = True)
	
	##
	x = T.exp(input)
	self.p_y_given_x = x/T.sum(x,axis=1,keepdims=True)
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.output = T.argmax(self.p_y_given_x, axis=1,keepdims=True)
        self.params = []
        self.input = input
	##
        self.gparams = []

    def negative_log_likelihood(self, y,y_wgt):
	# y is same to input [n,nclass,h,w]

	return -T.mean(T.log(self.p_y_given_x)*y_wgt*y)
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
	# y is [n,nc,h,w], the axis=1 is label
        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
	    y_label = T.argmax(y, axis=1,keepdims=True)
            return T.mean(T.neq(self.output, y_label))
        else:
            raise NotImplementedError()
