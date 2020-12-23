'''
lstmlayer.py
'''

import theano
import theano.tensor as T
import numpy as np
from build_w_b2 import build_w_b2
from kmaxcut import kmaxcut

## [Graves 2013: Generating sequences with recurrent neural networks]
class LSTMLayer1(object):
    #### input_shape: (tsteps,nsamples,feadim)
    #### hidden_dim
    #### hidden0/cell0: 1*nsamples*hidden_dim
    def __init__(self,input,input_shape,hidden_dim,hidden0,cell0):
	n1 = input_shape[2]
	n2 = hidden_dim
	w_xi,b_i = build_w_b2(n1,n2,'uniform')
	w_hi,_   = build_w_b2(n2,n2,'uniform')
	w_xj,b_j = build_w_b2(n1,n2,'uniform')
	w_hj,_   = build_w_b2(n2,n2,'uniform')
	w_xf,b_f = build_w_b2(n1,n2,'uniform')
	w_hf, _  = build_w_b2(n2,n2,'uniform')
	w_xo,b_o = build_w_b2(n1,n2,'uniform')
	w_ho, _  = build_w_b2(n2,n2,'uniform')
	
	g_w_xi = np.zeros_like(w_xi)
	g_w_hi = np.zeros_like(w_hi)
	g_w_xj = np.zeros_like(w_xj)
	g_w_hj = np.zeros_like(w_hj)
	g_w_xf = np.zeros_like(w_xf)
	g_w_hf = np.zeros_like(w_hf)
	g_w_xo = np.zeros_like(w_xo)
	g_w_ho = np.zeros_like(w_ho)
	g_b_i = np.zeros_like(b_i)
	g_b_j = np.zeros_like(b_j)
	g_b_f = np.zeros_like(b_f)
	g_b_o = np.zeros_like(b_o)
		
	self.w_xi = theano.shared(value=w_xi,name='w_xi',borrow=True)
	self.w_hi = theano.shared(value=w_hi,name='w_hi',borrow=True)
	self.w_xj = theano.shared(value=w_xj,name='w_xj',borrow=True)
	self.w_hj = theano.shared(value=w_hj,name='w_hj',borrow=True)
	self.w_xf = theano.shared(value=w_xf,name='w_xf',borrow=True)
	self.w_hf = theano.shared(value=w_hf,name='w_hf',borrow=True)
	self.w_xo = theano.shared(value=w_xo,name='w_xo',borrow=True)
	self.w_ho = theano.shared(value=w_ho,name='w_ho',borrow=True)
	self.b_i = theano.shared(value=b_i,name='b_i',borrow=True)	
	self.b_j = theano.shared(value=b_j,name='b_j',borrow=True)
	self.b_f = theano.shared(value=b_f,name='b_f',borrow=True)
	self.b_o = theano.shared(value=b_o,name='b_o',borrow=True)

	self.g_w_xi = theano.shared(value=g_w_xi,name='g_w_xi',borrow=True)
        self.g_w_hi = theano.shared(value=g_w_hi,name='g_w_hi',borrow=True)
        self.g_w_xj = theano.shared(value=g_w_xj,name='g_w_xj',borrow=True)
        self.g_w_hj = theano.shared(value=g_w_hj,name='g_w_hj',borrow=True)
        self.g_w_xf = theano.shared(value=g_w_xf,name='g_w_xf',borrow=True)
        self.g_w_hf = theano.shared(value=g_w_hf,name='g_w_hf',borrow=True)
        self.g_w_xo = theano.shared(value=g_w_xo,name='g_w_xo',borrow=True)
        self.g_w_ho = theano.shared(value=g_w_ho,name='g_w_ho',borrow=True)
        self.g_b_i = theano.shared(value=g_b_i,name='g_b_i',borrow=True)   
        self.g_b_j = theano.shared(value=g_b_j,name='g_b_j',borrow=True)
        self.g_b_f = theano.shared(value=g_b_f,name='g_b_f',borrow=True)
        self.g_b_o = theano.shared(value=g_b_o,name='g_b_o',borrow=True)

	self.params = [self.w_xi,self.w_hi,self.w_xj,self.w_hj,self.w_xf,self.w_hf,self.w_xo,self.w_ho,self.b_i,self.b_j,self.b_f,self.b_o]
	self.gparams = [self.g_w_xi,self.g_w_hi,self.g_w_xj,self.g_w_hj,self.g_w_xf,self.g_w_hf,self.g_w_xo,self.g_w_ho,self.g_b_i,self.g_b_j,self.g_b_f,self.g_b_o]
	
	#def dropout(x,isTraining):
	#    return T.switch(isTraining,x *T.cast(self.srng.binomial(size=x.shape,p=self.p), theano.config.floatX),self.p * x)

	def step(xt,htm1,ctm1,wxi,whi,wxj,whj,wxf,whf,wxo,who,bi,bj,bf,bo):
	    it = T.tanh(T.dot(xt,wxi)+T.dot(htm1,whi)+bi.dimshuffle('x',0))	
	    jt = T.nnet.sigmoid(T.dot(xt,wxj)+T.dot(htm1,whj)+bj.dimshuffle('x',0))
	    ft = T.nnet.sigmoid(T.dot(xt,wxf)+T.dot(htm1,whf)+bf.dimshuffle('x',0))
	    ot = T.dot(xt,wxo)+T.dot(htm1,who)+bo.dimshuffle('x',0)
	    ct = ctm1*ft + it*jt
	    ht = T.tanh(ct)*ot
            #ht = T.switch(isTr,ht *T.cast(srng.binomial(size=ht.shape,p=p), theano.config.floatX),p * ht)
	    return [ot,ht,ct]
	
	([output,hidden,cell],_) = theano.scan(step, sequences = input, outputs_info= [None,hidden0,cell0],\
           		                non_sequences = [self.w_xi,self.w_hi,self.w_xj,self.w_hj,\
							self.w_xf,self.w_hf,self.w_xo,self.w_ho,\
							self.b_i,self.b_j,self.b_f,self.b_o])
	self.output = output
	self.hidden = hidden
	self.cell = cell

## [Graves 2013: Generating sequences with recurrent neural networks] + predict
class LSTMLayer2(object):
    #### input_shape: (nsamples,feadim)
    #### hidden_dim
    #### hidden0/cell0: 1*nsamples*hidden_dim
    #### refdata: m*feadim
    def __init__(self,input,input_shape,hidden_dim,hidden0,cell0,refdata,knn_idx,k):
        n1 = input_shape[2]
        n2 = hidden_dim
	#print 'n1=',n1,'n2=',n2
        w_xi,b_i = build_w_b2(n1,n2,'uniform')
        w_hi,_   = build_w_b2(n2,n2,'uniform')
        w_xj,b_j = build_w_b2(n1,n2,'uniform')
        w_hj,_   = build_w_b2(n2,n2,'uniform')
        w_xf,b_f = build_w_b2(n1,n2,'uniform')
        w_hf, _  = build_w_b2(n2,n2,'uniform')
        w_xo,b_o = build_w_b2(n1,n2,'uniform')
        w_ho, _  = build_w_b2(n2,n2,'uniform')

        g_w_xi = np.zeros_like(w_xi)
        g_w_hi = np.zeros_like(w_hi)
        g_w_xj = np.zeros_like(w_xj)
        g_w_hj = np.zeros_like(w_hj)
        g_w_xf = np.zeros_like(w_xf)
        g_w_hf = np.zeros_like(w_hf)
        g_w_xo = np.zeros_like(w_xo)
	g_w_ho = np.zeros_like(w_ho)
        g_b_i = np.zeros_like(b_i)
        g_b_j = np.zeros_like(b_j)
        g_b_f = np.zeros_like(b_f)
        g_b_o = np.zeros_like(b_o)

        self.w_xi = theano.shared(value=w_xi,name='w_xi',borrow=True)
        self.w_hi = theano.shared(value=w_hi,name='w_hi',borrow=True)
        self.w_xj = theano.shared(value=w_xj,name='w_xj',borrow=True)
        self.w_hj = theano.shared(value=w_hj,name='w_hj',borrow=True)
        self.w_xf = theano.shared(value=w_xf,name='w_xf',borrow=True)
        self.w_hf = theano.shared(value=w_hf,name='w_hf',borrow=True)
        self.w_xo = theano.shared(value=w_xo,name='w_xo',borrow=True)
        self.w_ho = theano.shared(value=w_ho,name='w_ho',borrow=True)
        self.b_i = theano.shared(value=b_i,name='b_i',borrow=True)
        self.b_j = theano.shared(value=b_j,name='b_j',borrow=True)
        self.b_f = theano.shared(value=b_f,name='b_f',borrow=True)
        self.b_o = theano.shared(value=b_o,name='b_o',borrow=True)

        self.g_w_xi = theano.shared(value=g_w_xi,name='g_w_xi',borrow=True)
        self.g_w_hi = theano.shared(value=g_w_hi,name='g_w_hi',borrow=True)
        self.g_w_xj = theano.shared(value=g_w_xj,name='g_w_xj',borrow=True)
        self.g_w_hj = theano.shared(value=g_w_hj,name='g_w_hj',borrow=True)
        self.g_w_xf = theano.shared(value=g_w_xf,name='g_w_xf',borrow=True)
        self.g_w_hf = theano.shared(value=g_w_hf,name='g_w_hf',borrow=True)
        self.g_w_xo = theano.shared(value=g_w_xo,name='g_w_xo',borrow=True)
        self.g_w_ho = theano.shared(value=g_w_ho,name='g_w_ho',borrow=True)
        self.g_b_i = theano.shared(value=g_b_i,name='g_b_i',borrow=True)
        self.g_b_j = theano.shared(value=g_b_j,name='g_b_j',borrow=True)
        self.g_b_f = theano.shared(value=g_b_f,name='g_b_f',borrow=True)
        self.g_b_o = theano.shared(value=g_b_o,name='g_b_o',borrow=True)
        
	self.params = [self.w_xi,self.w_hi,self.w_xj,self.w_hj,self.w_xf,self.w_hf,self.w_xo,self.w_ho,self.b_i,self.b_j,self.b_f,self.b_o]
        self.gparams = [self.g_w_xi,self.g_w_hi,self.g_w_xj,self.g_w_hj,self.g_w_xf,self.g_w_hf,self.g_w_xo,self.g_w_ho,self.g_b_i,self.g_b_j,self.g_b_f,self.g_b_o]

        #def dropout(x,isTraining):
        #    return T.switch(isTraining,x *T.cast(self.srng.binomial(size=x.shape,p=self.p), theano.config.floatX),self.p * x)

        def step(xt,htm1,ctm1,wxi,whi,wxj,whj,wxf,whf,wxo,who,bi,bj,bf,bo,rdata,kidx,k):
            it = T.tanh(T.dot(xt,wxi)+T.dot(htm1,whi)+bi.dimshuffle('x',0))
            jt = T.nnet.sigmoid(T.dot(xt,wxj)+T.dot(htm1,whj)+bj.dimshuffle('x',0))
            ft = T.nnet.sigmoid(T.dot(xt,wxf)+T.dot(htm1,whf)+bf.dimshuffle('x',0))
            ot = T.dot(xt,wxo)+T.dot(htm1,who)+bo.dimshuffle('x',0)
            ct = ctm1*ft + it*jt
            ht = T.tanh(ct)*ot
	    ot = ot/T.sqrt((ot**2).sum(axis=1,keepdims = True))
	    dis = T.exp(2*T.dot(ot,rdata.T)-2)
	    weight = kmaxcut(dis,kidx,k)
	    weight = weight/T.sum(weight,axis=1,keepdims=True)
	    ot = T.dot(weight,rdata) 
            #ht = T.switch(isTr,ht *T.cast(srng.binomial(size=ht.shape,p=p), theano.config.floatX),p * ht)
            return [ot,ht,ct]

        ([output,hidden,cell],_) = theano.scan(step,outputs_info= [input,hidden0,cell0],\
                                        non_sequences = [self.w_xi,self.w_hi,self.w_xj,self.w_hj,\
                                                        self.w_xf,self.w_hf,self.w_xo,self.w_ho,\
                                                        self.b_i,self.b_j,self.b_f,self.b_o,refdata,knn_idx,k],n_steps=5)
        self.output = output
        self.hidden = hidden
        self.cell = cell
                             
