'''
conv layer
'''
import theano
import theano.tensor as T
import numpy as np
#from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from utis import build_w_b_kernel

class ConvLayer(object):
    ## filter_shape: [feamap,kernelH,kernelW]
    ## stride: [hstride,wstride]
    ## in_shape: [batch_size,channels,h,w]
    ## poosize: [poolH,poolW]
    def __init__(self,rng,input,layerparams):#filter_shape,conv_stride,in_shape,poolsize):
	self.input = input
	kernel = layerparams['kernel']
	in_shape = layerparams['in_shape']
	conv_stride = layerparams['conv_stride']
	conv_pad = layerparams['conv_pad']
	conv_sig = layerparams['conv_sig']

	##
	#W,b = build_w_b_kernel(rng,kernel,'gaussian',(0.0,0.01))
	W,b = build_w_b_kernel(rng,kernel,'uniform',1.) 
	self.W = theano.shared(W,borrow=True)
	self.b = theano.shared(b,borrow=True)
	##
	conv_out = conv2d( input = input, filters = self.W, filter_shape = kernel, \
				subsample=conv_stride, input_shape = in_shape,border_mode = conv_pad)
	RELU = lambda x: x*(x>0)
	#pool_in  = RELU()
	if conv_sig == 'RELU': 
	    self.output = RELU(conv_out + self.b.dimshuffle('x',0,'x','x'))
	elif conv_sig == 'sigmoid':
	    self.output = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x',0,'x','x'))
	elif conv_sig == 'tanh':
            self.output = T.tanh(conv_out + self.b.dimshuffle('x',0,'x','x'))
	elif conv_sig == 'linear':
            self.output = conv_out + self.b.dimshuffle('x',0,'x','x')
	elif conv_sig == 'exp_norm':
            self.output = conv_out + self.b.dimshuffle('x',0,'x','x')
	    self.output = T.exp(self.output-T.max(self.output,axis=[2,3],keepdims=True))
	    #self.output = self.output/T.max(self.output,axis=[2,3],keepdims=True)
	else:
	    assert(conv_sig=='RELU' or conv_sig=='sigmoid' or conv_sig=='tanh' or conv_sig=='linear' \
			or conv_sig =='exp_norm')

	self.params = [self.W, self.b]
	##
	self.gW = theano.shared(np.zeros_like(W),borrow = True)
	self.gb = theano.shared(np.zeros_like(b),borrow = True)
	self.gparams = [self.gW,self.gb]
	


