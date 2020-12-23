'''
runvgg19.py

use MISC.resize
'''

import numpy as np
import tensorflow as tf

import vgg19_tf
from vgg_utis import vgg_process_images, print_prob # -- 
from utis import read_image
import pdb
## paramters

gpu_id = '/gpu:3'
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


model_path = '/data/cuizhen/Data/Model/vgg19.npy'
batch_size = 1
vgg_out_layers = np.asarray((10,11,12,14,15,16,20))


img_param = {}
img_param['interp_tool'] = 'misc' # misc or skimage
img_param['interp'] = 'bilinear'
img_param['normal_hw'] = (224,224)
img_param['normal_type'] = 'keep_all_content' # 'keep_aspect_ratio' or 'keep_all_content'

##
fpath = "./test_data/tiger.jpeg"
img1 = read_image(fpath,True,True,-1)

fpath = "./test_data/puzzle.jpeg"
img2 = read_image(fpath,True,True,-1)

###################

#batch = vgg_process_images([img1,img2],224,224,normal_type='keep_aspect_ratio',interp = 'bilinear')# 'keep_all_content'
batch1 = vgg_process_images([img1],**img_param)# 'keep_all_content'
batch2 = vgg_process_images([img2],**img_param)# 'keep_all_content'

#vgg = vgg19_tf.Vgg19(batch_size, model_path)

#with tf.Session(
#        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device(gpu_id):

    #config = tf.ConfigProto(log_device_placement=True)
    #config.gpu_options.allow_growth = True

    vgg = vgg19_tf.Vgg19(model_path, vgg_out_layers)

    sess = tf.Session(config = config)

    feed_dict = {vgg.images: batch1}



    out_layers = sess.run(vgg.out_layers, feed_dict=feed_dict)
    for ii, x in enumerate(out_layers):
         print('{}--{}'.format(ii,x.shape))
    for ii in range(out_layers[1].shape[-1]):
        print(out_layers[1][0,:,:,ii])
        #pdb.set_trace()
    x = out_layers[1][0]
    print(np.sum(x**2,axis=-1))
    print_prob(out_layers[-1][0], './synset.txt')

    #feed_dict = {vgg.images: batch2}

    #prob2 = sess.run(vgg.prob, feed_dict=feed_dict)
    #print(prob2)
    #print_prob(prob2[0], './synset.txt')

