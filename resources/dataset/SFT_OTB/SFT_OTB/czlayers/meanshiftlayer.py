'''
meanshiftlayer.pu
'''
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

class MeanShiftLayer(object):
    ## filter_shape: [feamap,kernelH,kernelW]
    ## stride: [hstride,wstride]
    ## in_shape: [batch_size,channels,h,w]
    ## poosize: [poolH,poolW]
    def __init__(self,rng,input,filter_shape,in_shape):
	self.input = input
 	W_bound = 1. / np.prod(filter_shape)
	##
	W = np.zeros(filter_shape,dtype = theano.config.floatX)
	W[:,:,:,:] = W_bound
	self.W = theano.shared(W,borrow=True)
	##
	self.output = conv2d( input = input, filters = self.W, filter_shape = filter_shape, subsample=(1,1), input_shape = in_shape)
	##
	self.params = []
	self.gparams = []
	


