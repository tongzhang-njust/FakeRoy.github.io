'''
for recurrent shape regression
runL2.py
'''

import os,sys,pdb
import numpy as np
import tensorflow as tf

from Logger import *
from utis import *
from funs_trackingK4_corner import *

## vgg
from vgg_utis import vgg_process_images, vgg_resize_maps
import vgg19_tf2
## graph
from graph import grid_graph, fea_graph,laplacian
## cgcnn
#from models import cgcnn
#from perf import fit, evaluate
from models import graphtracker# import GTTr, GTTe
from resp_test import resp_newton

#########################
##### gpu parameter #####
#########################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_id = '/gpu:3'
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


#########################
#### data params ########
#########################

data_path = r'/mnt/cuizhen/Data/VOT2016_process/'
cache_path = r'/mnt/cuizhen/Results/VOT2016_result_20171110/'
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)
pstr = 'gcnn'

##

start_sample = 0#0
end_sample = 60#97
step_sample =1
idx_fdrs = np.arange(start_sample,end_sample,step_sample)
#idx_fdrs.reverse()
print("{}".format(idx_fdrs))

####
padding = {'large':1,'height':0.4,'generic':2} # 25~50: 2.5 others 2.2
cell_size0 = 4  ##
batch_size = 1 # fixed
max_win2 = 1600
min_win2 = 1600
fea_sz = np.asarray([57,57])

#########################
####### VGG Model #######
#########################

vgg_model_path = '/home/caiyouyi/Data/Model/vgg19.npy'
vgg_batch_size = 1
vgg_out_layers =  [1,3, 10,12,14,16]#np.asarray((10,11,12,14,15,16))
#vgg_out_layers2 = [1,3]# np.asarray((1)) # scale


vgg_is_lrn = False

## image processing params for vgg
img_param = {}
img_param['interp_tool'] = 'misc' # misc or skimage
img_param['interp'] = 'bilinear'
img_param['normal_hw'] = (224,224)
img_param['normal_type'] = 'keep_all_content'

##################################
###### graph parameters ##########
##################################

gh1 = {'height_width':None,'number_edges':4,'metric':'euclidean','normalized_laplacian': True}

#gh2 = {'height_width':None,'number_edges':0.1,'metric':'euclidean','normalized_laplacian': True}
k_step = 2.0

#### pca params
pca_flag = False
pca_is_mean = True
pca_is_norm = False
pca_energy = 100
####
#nn_p = 6
#nn_K = 20
nn_gamma = np.float32(1.0)
vgg_map_conv1=64
####################### cf params ###############################

#kernel_sigma = 0.5
#kernel_type = 'linear'
#kernel_gamma = np.float32(1.) #1.e-6
update_factor = 0.0075 # Jogging learning 0.005, others 0.015
cf_nframe_update = 1
#weight_update_factor = 0.01

#### scale params
#update_factor_s=0.01
sparam_nScales = 7
sparam_scale_step = 1.02
sparam_angles = np.float32(np.asarray([-10,-5.,0,5.,10])/180.0*np.pi)
sparam_nAngles = len(sparam_angles)
#sparam_angles = np.reshape(sparam_angles,(-1,1))
#sparam_scale_model_max_area = 512
#sparam_scale_sigma_factor = 1#0.25
#sparam_K_ratio = 1.0
#sparam_m_type = 'expert' # 'dot' or 'expert' ??
#sparam_gamma = nn_gamma
#lamda=1e-2
#scale_model_sz = [224,224] #[w,h]
