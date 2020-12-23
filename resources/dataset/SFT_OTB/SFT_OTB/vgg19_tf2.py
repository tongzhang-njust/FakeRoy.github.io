import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    def __init__(self, vgg19_npy_path=None, inds_outlayers = [19], nAngle = 5,  inds_outlayers2 = None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

        self.images = tf.placeholder("float", [None, None, None, 3])
        self.norm_h = tf.placeholder("int32")
        self.norm_w = tf.placeholder("int32")


        with tf.name_scope("content_vgg"):
            self.build(self.images)
        
        self.get_fea_layers(inds_outlayers,inds_outlayers2)
        self.vgg_fea = self.resize_map(self.out_layers,self.norm_h,self.norm_w)

        if inds_outlayers2 is not None:
            self.norm_h2 = tf.placeholder("int32")
            self.norm_w2 = tf.placeholder("int32")
            self.vgg_fea2 = self.resize_map(self.out_layers2,self.norm_h2,self.norm_w2)

        ## rotation, scale
        self.imgs = tf.placeholder("float32", [None, None, None, 3]) # [n, h, w, 3]
        self.img_sz = tf.placeholder("float32",[None]) # [h,w]
        self.angles = tf.placeholder("float32", [nAngle]) # [nangle]
        #self.angle_seg = tf.placeholder("int32",[None]) #
        self.nangle = nAngle
        self.angle1 = tf.placeholder("float32",[1])
        self.bbx  = tf.placeholder("float32",[None, 4]) # [ncrop, 4]; [h1,w1,h2,w2]
        self.bind = tf.placeholder("int32", [None])   # [ncrop]
        self.crop_size = tf.placeholder("int32",[None]) # [h_nm, w_nm]
        #self.nAngle = 5

        self.crop_resize_with_rotation_scaling()
        self.crop_resize_with_rotation_scaling1()
        
    def crop_resize_with_rotation_scaling1(self):

        # [n,h,w,c]
        self.transform1 = tf.contrib.image.angles_to_projective_transforms(self.angle1[0], self.img_sz[0], self.img_sz[1])
        image = tf.contrib.image.transform(self.imgs, self.transform1)  # [n,h,w,c]
        self.patches1 = tf.image.crop_and_resize(image, self.bbx, self.bind, self.crop_size) # [npatches, nmh, nmw, c]


    def crop_resize_with_rotation_scaling(self):

        # [n,h,w,c]
        self.patches = []
        self.transforms = []
        angles = tf.split(self.angles, self.nangle)
        #self.angles2 = angles
        for al in angles: # 
            transform = tf.contrib.image.angles_to_projective_transforms(al[0], self.img_sz[0], self.img_sz[1])
            image = tf.contrib.image.transform(self.imgs, transform)  # [n,h,w,c]
            patches = tf.image.crop_and_resize(image, self.bbx, self.bind, self.crop_size) # [npatches, nmh, nmw, c]
            self.patches.append(patches) 
            self.transforms.append(transform)
        self.patches = tf.stack(self.patches, axis=0) # [al, sl, nmh, nmw, c]
        
        ## return patches, transforms, offset
            

    def get_fea_layers(self, inds_outlayers, inds_outlayers2):
        
        all_layers = [self.conv1_1,self.conv1_2, self.conv2_1,self.conv2_2, \
                      self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4,\
                      self.conv4_1,self.conv4_2, self.conv4_3, self.conv4_4,\
                      self.conv5_1,self.conv5_2, self.conv5_3, self.conv5_4]
        #keywords = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv3_4','conv4_1','conv4_2','conv4_3',\
        #                'conv4_4','conv5_1','conv5_2','conv5_3','conv5_4','fc6','fc7','fc8','prob']
        self.mapnums  = np.asarray((64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096,4096,1000,1000))

        self.out_layers = []
        self.out_num_maps = []
        for ix in inds_outlayers:
            #if isLRN == False:
            #    self.outlayers.append(self.vggnet[keywords[ix-1]])
            #else:
            self.out_layers.append(all_layers[ix-1]) 
            self.out_num_maps.append(self.mapnums[ix-1])

         ## 
        self.out_num_maps = np.asarray(self.out_num_maps)
        x = np.cumsum(self.out_num_maps)
        self.out_map_nlayer = len(x)
        self.out_map_idx = np.zeros(self.out_map_nlayer + 1,dtype=np.int32)
        self.out_map_idx[1:] = x
        self.out_map_total = x[-1]

        if inds_outlayers2 is not None:
            self.out_layers2 = []
            self.out_num_maps2 = []
            for ix in inds_outlayers2:
                self.out_layers2.append(all_layers[ix-1])
                self.out_num_maps2.append(self.mapnums[ix-1])

            self.out_num_maps2 = np.asarray(self.out_num_maps2)
            x = np.cumsum(self.out_num_maps2)
            self.out_map_nlayer2 = len(x)
            self.out_map_idx2 = np.zeros(self.out_map_nlayer2 + 1,dtype=np.int32)
            self.out_map_idx2[1:] = x
            self.out_map_total2 = x[-1]

    def resize_map(self,out_layers,norm_h,norm_w):
        n = out_layers[0].shape[0] 
        vgg_fea=[]
        for hmap in out_layers:
            x=tf.image.resize_images(hmap,[norm_h,norm_w])
            vgg_fea.append(x)
        return tf.concat(vgg_fea,3)
        

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb # rgb * 255.0 ??

        # Convert RGB to BGR
        red, green, blue = tf.split( rgb_scaled,3, 3)
        #assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat( [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ],3)
        #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        #self.fc6 = self.fc_layer(self.pool5, "fc6")
        #assert self.fc6.get_shape().as_list()[1:] == [4096]
        #self.relu6 = tf.nn.relu(self.fc6)

        #self.fc7 = self.fc_layer(self.relu6, "fc7")
        #self.relu7 = tf.nn.relu(self.fc7)

        #self.fc8 = self.fc_layer(self.relu7, "fc8")

        #self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
