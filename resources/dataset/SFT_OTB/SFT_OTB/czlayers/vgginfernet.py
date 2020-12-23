# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

import lasagne
from lasagne.utils import floatX
from netmodels import *

import pickle
import numpy as np
import skimage.transform
import scipy.misc as MISC
#from PIL import Image
import time
import theano.typed_list
import theano.tensor as T

class VGGInferNet(object):
    def __init__(self,model_file,net_type,inds_outlayers,isLRN):
        if net_type == 'vgg19':
            net, prob = build_model_vgg19()
            model = pickle.load(open(model_file))
            self.CLASSES = model['synset words']
            self.mean_value = floatX(model['mean value'])
	    
	    #print self.mean_value
            #self.MEAN_IMAGE = np.zeros((3,224,224),dtype=np.float32)
            #self.MEAN_IMAGE[0,:,:] = mn[0]
            #self.MEAN_IMAGE[1,:,:] = mn[1]
            #self.MEAN_IMAGE[2,:,:] = mn[2]

	    ## 1~19 layer + prob layer
            keywords = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv3_4','conv4_1','conv4_2','conv4_3',\
                        'conv4_4','conv5_1','conv5_2','conv5_3','conv5_4','fc6','fc7','fc8','prob']
            self.mapnums  = np.asarray((64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096,4096,1000))

	    ##
            lasagne.layers.set_all_param_values(prob, model['param values'])
            self.vggnet = net
	   
	    ##
            self.outlayers = []
            for ix in inds_outlayers:
                if isLRN == False:
                    self.outlayers.append(self.vggnet[keywords[ix-1]])
                else:
                    ss = keywords[ix-1]
                    ss = 'lrn'+ss[-3:]
                    self.outlayers.append(self.vggnet[ss])
	  
            x   = T.ftensor4('x')
            out = lasagne.layers.get_output(self.outlayers, x, deterministic=True)
	    #out = [lasagne.layers.LocalResponseNormalization2DLayer(iout) for iout in out]
	    #def lrn(fea):
	    #	return lasagne.layers.LocalResponseNormalization2DLayer(fea)
	    #out, _ = theano.map(fn=lrn,sequences=out)
            self.f = theano.function([x],out) 
        else:
            print('error:{}'.format(net_type))

    #### preprocessing
    def prep_image(self,im,normal_height,normal_width,normal_type,is_swap_axis):
        h, w, _ = im.shape
        if normal_type == 'keep_aspect_ratio':
            r1 = 1.*normal_height/h
            r2 = 1.*normal_width/w
            r  = np.maximum(r1,r2)
            if r != 1.:
            	im = MISC.imresize(im,r,interp='bilinear')
    	    # Central crop
            h, w, _ = im.shape
            im = im[h//2-normal_height//2:h//2+normal_height//2, w//2-normal_width//2:w//2+normal_width//2]
        elif normal_type == 'keep_all_content':
            if h!=normal_height or w!=normal_width:
                im = MISC.imresize(im,[normal_height,normal_width],interp='bilinear')
        else:
            print('normal_type error, please set <keep_aspect_ratio> or <keep_all_content>.')

        if is_swap_axis:
    	    # Shuffle axes to c01
    	    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    	    # Convert to BGR
    	    im = im[::-1, :, :]
        return im #floatX(im[np.newaxis])

    #### 
    # 'x': h*w*c
    def get_output(self,im,normal_height,normal_width,normal_type):
	## preprocessing
        im = floatX(self.prep_image(im,normal_height,normal_width,normal_type,True))
	##	
        im[0,:,:] = im[0,:,:] - self.mean_value[0]
        im[1,:,:] = im[1,:,:] - self.mean_value[1]
        im[2,:,:] = im[2,:,:] - self.mean_value[2]
        im = im[np.newaxis]
        data_outlayers = self.f(im)#lasagne.layers.get_output(self.outlayers, im, deterministic=True)
        return data_outlayers,self.CLASSES

    ####
    ## ims: n*h*w*c
    def get_output_batch(self,ims,normal_height,normal_width,normal_type):
        ## preprocessing
        flag = type(ims) is list
        if flag:
            n = len(ims)
            c = ims[0].shape[-1]
        else:
            n, _, _, c = ims.shape
        ims_out = np.zeros((n,c,normal_height,normal_width),dtype = np.float32)
        for ii in xrange(n):
            im = self.prep_image(ims[ii],normal_height,normal_width,normal_type,True)
            ims_out[ii,:,:,:]  = im
	##
        ims_out[:,0,:,:] = ims_out[:,0,:,:] - self.mean_value[0]
        ims_out[:,1,:,:] = ims_out[:,1,:,:] - self.mean_value[1]
        ims_out[:,2,:,:] = ims_out[:,2,:,:] - self.mean_value[2]
	
        data_outlayers = self.f(ims_out)
        return data_outlayers,self.CLASSES


