'''
kmaxcut.py
'''

""" Ops for downsampling images.

Planned:
DownsampleFactorMax, DownsampleAvg, DownsampleSoftmax.

"""
# This file should move along with conv.py
import __builtin__

import numpy as np
import theano
from theano import gof, Op, Variable, Apply
import theano.tensor as T
from theano.gradient import grad_undefined

# W: n_lmks*2*9; float32
# xg: (n_std_width*n)*2*h*w; float32
# xs/xo: 2*n*2; float32
# plmks/olmks: n*(n_lmks*2); float32
# bw: n*1; float32
# sbw/sbwi: 1; int32
# plidx: 1024*4,[oi,oj,i,j]; int32
# ncells(4): 1; int32
# nblocks(4): 1; int32
# output >> : n*n_lmks*n_blocks*(n_cells*npros)

def kmaxcut(x,knn_idx,k):
    op = KMAXCUT()
    output = op(x,knn_idx,k)
    return output


class KMAXCUT(Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self,x,knn_idx,k):
	
        x = T.as_tensor_variable(x)
	knn_idx = T.as_tensor_variable(knn_idx)
	k = T.as_tensor_variable(k)
	bcast = (False,False)
        return gof.Apply(self,inputs = [x,knn_idx,k], outputs = [T.tensor(theano.config.floatX,bcast)])

    def perform(self, node, inp, out):
        x, knn_idx, k = inp
        #z = out
	# get attributes
	n1 = x.shape[0]
	n2 = x.shape[1]
	n11 = knn_idx.shape[0]
	k1  = knn_idx.shape[1]
	assert n11 == n1
	assert k == k1
	# verify output shape
        z_shape = (n1,n2)
	z = np.zeros(z_shape,dtype=np.float32)
	x2 = np.copy(x)
	for i in xrange(k):
	    idx = np.argmax(x2,axis=1)
	    for j in xrange(n1):
		m = idx[j]
	    	knn_idx[j,i] = m
	    	x2[j,m] = -1.0e9 
	   	z[j,m]  = x[j,m]
	out[0][0]=z
    '''
    def infer_shape(self, node, in_shapes):
	x,knn_idx,k = node.inputs
	x_shape,knn_idx_shape,k_shape = in_shapes
        shp = (x_shape[0],x_shape[1])
        return [shp]
    '''
    def grad(self, inp, grads):

        x,knn_idx,k = inp
	gz, = grads
	
        return [KMAXCUTGrad()(x,knn_idx,k,gz),\
	        grad_undefined(self, 1, knn_idx),\
		grad_undefined(self, 2, k)]
    '''	
    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        if self.mode != 'max':
            raise theano.gof.utils.MethodNotDefined()
        x0,x1 = inp
        z, = out
        fail = sub['fail']
        return """
	int typenum = PyArray_ObjectType((PyObject*)%(x0)s, 0);
        int z_r, z_c; // shape of the output
        if(PyArray_NDIM(%(x0)s)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 2d ndarray");
            %(fail)s;
        }
        r = PyArray_DIMS(%(x0)s)[0];
        c = PyArray_DIMS(%(x1)s)[1];

	z_r = r;
	z_c = c;
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=2
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x0)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x0)s)[1])
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[2] = {0,0,0,0};
          dims[0]=z_r;
          dims[1]=z_c;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(2, dims, typenum,0);
        }

        // used for indexing a pool region inside the input
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x0)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x0)s)[1]; k++){
		  z[b]
              }
            }
        }
        """ % locals()
        '''

class KMAXCUTGrad(Op):

    def make_node(self,x,knn_idx,k,gz):
	x = T.as_tensor_variable(x)
        knn_idx = T.as_tensor_variable(knn_idx)
        k = T.as_tensor_variable(k)
	gz = T.as_tensor_variable(gz)
        bcast = (False,False)
        return Apply(self, inputs = [x,knn_idx,k, gz], outputs = [T.tensor(theano.config.floatX,bcast)])

    def perform(self, node, inp, out):
	x,knn_idx,k,gz = inp
	n1 = gz.shape[0]
	n2 = gz.shape[1]
	z = np.zeros((n1,n2),dtype=np.float32)
	
	for i in xrange(k):
	    for j in xrange(n1):
		m = knn_idx[j,i]
		z[j,m] = np.sign(x[j,m])*gz[j,m]
	out[0][0]=z


