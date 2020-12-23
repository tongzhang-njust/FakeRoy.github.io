'''
conv + pooling layer
'''
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from utis import build_w_b_kernel

class ConvPoolLayer(object):
    ## filter_shape: [feamap,kernelH,kernelW]
    ## stride: [hstride,wstride]
    ## in_shape: [batch_size,channels,h,w]
    ## poosize: [poolH,poolW]
    def __init__(self,rng,input,layerparams):#filter_shape,conv_stride,in_shape,poolsize):
	self.input = input
	kernel = layerparams['kernel']
	in_shape = layerparams['in_shape']
	conv_stride = layerparams['conv_stride']
	pool_stride = layerparams['pool_stride']
	conv_pad = layerparams['conv_pad']
	pool_size = layerparams['pool_size']
	pool_pad = layerparams['pool_pad']

	##
	#W,b = build_w_b_kernel(rng,kernel,'gaussian',(0.0,0.01))
	W,b = build_w_b_kernel(rng,kernel,'uniform',1.) 
	self.W = theano.shared(W,borrow=True)
	self.b = theano.shared(b,borrow=True)
	##
	print 'in_shape',in_shape
	conv_out = conv2d( input = input, filters = self.W, filter_shape = kernel, \
				subsample=conv_stride, input_shape = in_shape,border_mode = conv_pad)
	RELU = lambda x: x*(x>0)
	#pool_in  = RELU()
	##
	pool_out = T.signal.pool.pool_2d(input = conv_out, ds = pool_size,st=pool_stride, padding=pool_pad,ignore_border = True, mode='max')
	#pool_out = downsample.max_pool_2d(input = conv_out, ds = poolsize, ignore_border = True)
	## 
	self.output = RELU(pool_out + self.b.dimshuffle('x',0,'x','x'))
	self.params = [self.W, self.b]
	##
	self.gW = theano.shared(np.zeros_like(W),borrow = True)
	self.gb = theano.shared(np.zeros_like(b),borrow = True)
	self.gparams = [self.gW,self.gb]
	


