'''
logreglayer.py
'''

import numpy as np
import theano
import theano.tensor as T
from utis import build_w_b

class LogRegLayer(object):

    def __init__(self, rng, input, layerparams):#input_shape, out_dim):
	in_shape = layerparams['in_shape']
	out_dim  = layerparams['class_num']
	#W,b = build_w_b(rng,in_shape[1],out_dim,'gaussian',(0.0,0.01))
        W,b = build_w_b(rng,in_shape[1],out_dim,'uniform',1.)
        self.W = theano.shared(W,borrow = True)
        self.b = theano.shared(b,borrow = True)
	
	##
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input
	##
	self.gW = theano.shared(np.zeros_like(W),borrow = True)
        self.gb = theano.shared(np.zeros_like(b),borrow = True)
        self.gparams = [self.gW,self.gb]

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
