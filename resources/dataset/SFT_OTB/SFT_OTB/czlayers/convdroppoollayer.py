'''
conv + dropout + pooling layer
'''
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from utis import build_w_b_kernel

class ConvDropPoolLayer(object):
    ## filter_shape: [feamap,kernelH,kernelW]
    ## stride: [hstride,wstride]
    ## in_shape: [batch_size,channels,h,w]
    ## poosize: [poolH,poolW]
    def __init__(self,rng,input,isTraining,layerparams):#filter_shape,conv_stride,in_shape,poolsize):
	self.input = input
	kernel = layerparams['kernel']
	in_shape = layerparams['in_shape']
	conv_stride = layerparams['conv_stride']
	pool_stride = layerparams['pool_stride']
	conv_pad = layerparams['conv_pad']
	pool_size = layerparams['pool_size']
	pool_pad = layerparams['pool_pad']
	srng = T.shared_randomstreams.RandomStreams(seed=12345)
	r =  np.float32(1.-layerparams['drop_rate'])

	##
	#W,b = build_w_b_kernel(rng,kernel,'gaussian',(0.0,0.01))
	W,b = build_w_b_kernel(rng,kernel,'uniform',1.) 
	self.W = theano.shared(W,borrow=True)
	self.b = theano.shared(b,borrow=True)
	##
	#print 'in_shape',in_shape
	conv_out = conv2d( input = input, filters = self.W, filter_shape = kernel, \
				subsample=conv_stride, input_shape = in_shape,border_mode = conv_pad)
	RELU = lambda x: x*(x>0)
	##
	conv_out = T.switch(isTraining,\
		conv_out *T.cast(srng.binomial(size=conv_out.shape,p=r), theano.config.floatX),r * conv_out)
	#mask = srng.binomial(size=conv_out.shape,p=r)
	#mask = T.cast(mask,theano.config.floatX)
	#conv_out = T.switch(isTraining,conv_out * mask,r * conv_out)
	#conv_out = T.cast(conv_out,theano.config.floatX)
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
	


