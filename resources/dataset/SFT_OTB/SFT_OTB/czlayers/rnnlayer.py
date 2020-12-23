'''
rnn.py
'''

import theano 
import theano.tensor as T
import numpy as np
from utis import build_w_b

class RNNLayer(object):
    #### input_shape: (nsamples,nmaps,h(or tsteps),w)-->(tsteps,nsamples,nmaps*w)
    #### hidden_dim
    #### out_shape: (nsamples,1,tsteps,hidden_dim)
    def __init__(self,rng,input,hidden0,layerparams):#input_shape,hidden_dim,hidden0):
	self.input = input
	in_shape = layerparams['in_shape']
	hid_dim   = layerparams['hid_dim']
	## 
	in_dim = in_shape[1]*in_shape[3]
	#W_ih,b_ih = build_w_b(rng,in_dim,hid_dim,'gaussian',(0.,0.01))
	W_ih,b_ih = build_w_b(rng,in_dim,hid_dim,'uniform',1.)
	self.W_ih = theano.shared(W_ih,borrow=True)
	self.b_ih = theano.shared(b_ih,borrow = True)
	##
	W_hh,_  = build_w_b(rng,hid_dim,1,'identity',0.001)
	self.W_hh = theano.shared(W_hh,borrow=True)
	#self.b_hh = theano.shared(np.zeros((hidden_shape[1],),dtype=theano.config.floatX),borrow = True)
	## -->	
	input = input.dimshuffle(2,0,1,3) # (n,c,h,w=1) --> (h,n,c,w=1)
        input = input.reshape((in_shape[2],in_shape[0],in_shape[1]*in_shape[3])) 
	##
	RELU = lambda x: x*(x>0)
	## 
	def step(x_t,h_tm1):
	    h_t = T.dot(x_t,self.W_ih) + T.dot(h_tm1,self.W_hh) + self.b_ih.dimshuffle('x',0)
	    return h_t
	##
	hiddens, _ = theano.scan(step,sequences=input,\
		 outputs_info = hidden0)
        #T.zeros_like( np.zeros((input_shape[1],hidden_dim),dtype=theano.config.floatX) ))
	##
	self.output = (RELU(hiddens)).dimshuffle(1,'x',0,2)	
	self.params  = [self.W_ih,self.b_ih,self.W_hh]
	##
        self.gW_ih = theano.shared(np.zeros_like(W_ih),borrow = True)
        self.gb_ih = theano.shared(np.zeros_like(b_ih),borrow = True)
	self.gW_hh = theano.shared(np.zeros_like(W_hh),borrow = True)
        self.gparams = [self.gW_ih,self.gb_ih,self.gW_hh]

