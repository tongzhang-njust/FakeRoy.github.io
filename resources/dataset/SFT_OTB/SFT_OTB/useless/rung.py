'''
run.py
'''

import sys,os
import numpy as np
import tensorflow as tf
import time
import pdb
from graph import grid_graph,replace_random_edges,laplacian
from coarsen import coarsen,perm_data
from models import cgcnn
from perf import fit, evaluate

#### data path
dir_data = os.path.join('..','data','mnist')

#### Graph parameters
gh = {'number_nodes':28,'number_edges':8,'metric':'euclidean','normalized_laplacian': True,\
	 'coarsening_levels':4, 'noise_random_edges':0}

#### feature graph
t_start = time.process_time()
#pdb.set_trace()
A = grid_graph(gh,corners=False)
A = replace_random_edges(A,gh['noise_random_edges'])
#### coarsening 
graphs, perm = coarsen(A,gh['coarsening_levels'],self_connections=False)
L = [laplacian(A) for A in graphs]
print('Execution time: {:.2f}'.format(time.process_time()-t_start))
del A

################
####  Data #####
################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dir_data, one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels

t_start = time.process_time()
train_data = perm_data(train_data, perm)
val_data =   perm_data(val_data, perm)
test_data =  perm_data(test_data, perm)
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
del perm
print('data shape: {}, {}, {}, label shape: {}, {}, {}.'.format(train_data.shape,val_data.shape,test_data.shape, train_labels.shape, val_labels.shape, test_labels.shape ))


###########################
#### Neural Networks ######
###########################

nn = {}
nn['dir_name']      = '/data/cuizhen/Results/mnist'
nn['num_epochs']    = 20
nn['batch_size']    = 128
nn['decay_step']   = mnist.train.num_examples/nn['batch_size']
nn['eval_frequency']= 30*nn['num_epochs']
nn['brelu']         = 'b1relu'
nn['pool']          = 'mpool1'
C    = max(mnist.train.labels) + 1

nn['regularization'] = 5e-4
nn['dropout']        = 0.5
nn['learning_rate']  = 0.02
nn['decay_rate']     = 0.95
nn['momentum']       = 0.9

nn['F'] = [32, 64] # signal lengths of each layer
nn['K'] = [25, 25] # poly orders
nn['p'] = [4,   4] # pooling size
nn['M'] = [512, C] # hidden numbers of last layers

nn['filter'] = 'chebyshev5'

if not os.path.isdir(nn['dir_name']):
    os.mkdir(nn['dir_name'])

with tf.device('/gpu:0'):
    #configure the gpu use
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #### actual model
    mdl = cgcnn.cgcnn(L,**nn)


    ####
    fit_accuracies, fit_losses, fit_time = fit(config, mdl, train_data,train_labels,val_data,val_labels)

    string, train_accuracy, train_f1, train_loss = evaluate(mdl, train_data, train_labels)
    print('train {}'.format(string))

    string, test_accuracy, test_f1, test_loss = evaluate( mdl, test_data, test_labels)
    print('test {}'.format(string))
















