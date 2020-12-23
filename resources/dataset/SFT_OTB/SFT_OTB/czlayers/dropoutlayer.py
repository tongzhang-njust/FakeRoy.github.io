'''
   dropoutlayer.py
'''
import theano
import theano.tensor as T
import numpy as np
import theano.tensor.shared_randomstreams as RD

## [Graves 2013: Generating sequences with recurrent neural networks]
class DropoutLayer(object):
    def __init__(self,r):
	self.srng = RD.RandomStreams(seed=12345)
	self.p    = 1-r

    def build(self,x,isTraining):
    	return T.switch(isTraining,x *T.cast(self.srng.binomial(size=x.shape,p=self.p), theano.config.floatX),self.p * x)
	'''
	if isTraining:
	    mask = self.srng.binomial(size=x.shape,p=self.p)
	    x = x * T.cast(mask, theano.config.floatX)
	else:
	    x = self.p * x
	return x
	'''

