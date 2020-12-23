'''
alignlayer.py
alignment of seqs in RNN.
'''

import theano
import theano.tensor as T
import numpy as np
from utis import build_w_b

class RNNAlignLayer(object):
    #### input_shape: (batch_size,1,time_step,feadim)
    #### out_shape: (batch_size,feadim)
    def __init__(self,rng,input,mask_seq,layerparams):#hidden,hidden_shape,mask_seq):
        in_shape = layerparams['in_shape']
        hid_dim = layerparams['hid_dim']

	##
	#W_io,b_io = build_w_b(rng,in_shape[3],hid_dim,'gaussian',(0.,.01))
        W_io,b_io = build_w_b(rng,in_shape[3],hid_dim,'uniform',1.)
        self.W_io = theano.shared(W_io,borrow=True)
        self.b_io = theano.shared(b_io,borrow = True)

	#v,b_v = build_w_b(rng,hid_dim,1,'gaussian',(0.,.01))
        v,b_v = build_w_b(rng,hid_dim,1,'uniform',1.)
        self.v  = theano.shared(v,borrow=True)
        self.b_v = theano.shared(b_v,borrow = True)

        ##
        alpha = T.exp(T.dot(T.tanh(T.dot(input,self.W_io) + \
			self.b_io.dimshuffle('x','x','x',0)),self.v)+self.b_v.dimshuffle('x','x','x',0))#(n,1,t,1)
        alpha = alpha*mask_seq
        alpha = T.Rebroadcast((0,False),(1,False),(2,False),(3, True))(alpha/alpha.sum(axis=2,keepdims=True))
        self.output = (input*alpha).sum(axis=(1,2))
        self.params = [self.W_io,self.b_io,self.v,self.b_v]
        ##
        self.gW_io = theano.shared(np.zeros_like(W_io),borrow = True)
        self.gb_io = theano.shared(np.zeros_like(b_io),borrow = True)
        self.gv = theano.shared(np.zeros_like(v),borrow = True)
	self.gb_v = theano.shared(np.zeros_like(b_v),borrow = True)
        self.gparams = [self.gW_io,self.gb_io,self.gv,self.gb_v]



class RNNAlignLayer2(object):
    #### input_shape: (batch_size,time_step,feadim1)
    #### hidden_shape: (batch_size,time_step,feadim2)
    #### out_shape: (out_dim)
    def __init__(self,rng,input,input_shape,hidden,hidden_shape,mask_seq):
        assert input_shape[0] == hidden_shape[0]
        assert input_shape[1] == hidden_shape[1]
        self.input = T.concatenate([hidden,hidden],axis=2) # ???
	#print input_shape,hidden_shape
	self.in_dim = hidden_shape[2]+hidden_shape[2]
	#self.in_dim = input_shape[2]+hidden_shape[2]
	self.out_dim = self.in_dim
	
	##
	'''
	mask_seq = T.Rebroadcast((0,False),(1,False),(2, True))(mask_seq/mask_seq.sum(axis=(1,2),keepdims=True))
	self.output = (self.input*mask_seq).sum(axis=1)
	self.output = self.output/T.sqrt(T.sum(self.output**2,axis=1,keepdims=True))
	self.params =[]
	self.gparams =[]
	'''
        ##
	
        W_bound = np.sqrt(6. / (self.in_dim + self.out_dim))
	W_io = np.asarray(rng.uniform(low=-W_bound,high=W_bound,size=(self.in_dim,self.out_dim)),dtype = theano.config.floatX)
        self.W_io = theano.shared(W_io,borrow=True)
	b_io = np.zeros((self.out_dim,),dtype=theano.config.floatX)
        self.b_io = theano.shared(b_io,borrow = True)
        W_bound = np.sqrt(6./(1.*self.out_dim))
	v = np.asarray(rng.uniform(low=-W_bound,high=W_bound,size=(self.out_dim,1)),dtype=theano.config.floatX)
        self.v  = theano.shared(v,borrow=True)
        ##
        alpha = T.exp(T.dot(T.tanh(T.dot(self.input,self.W_io) + self.b_io.dimshuffle('x','x',0)),self.v))
	alpha = alpha*mask_seq
        alpha = T.Rebroadcast((0,False),(1,False),(2, True))(alpha/alpha.sum(axis=(1,2),keepdims=True))
        self.output = (self.input*alpha).sum(axis=1)
        self.params = [self.W_io,self.b_io,self.v]
	##
	self.gW_io = theano.shared(np.zeros_like(W_io),borrow = True)
        self.gb_io = theano.shared(np.zeros_like(b_io),borrow = True)
        self.gv = theano.shared(np.zeros_like(v),borrow = True)
        self.gparams = [self.gW_io,self.gb_io,self.gv]
		
