'''
perf.py
'''

import numpy as np
import tensorflow as tf
import time, os, collections, shutil
import sklearn
from tfutis import _get_session
from utis import get_path

def fit(config, model, L, train_data, train_labels, val_data = None, val_labels = None):
    # model: graph model
    # L: n*hw*hw
    # train_data: n*hw*d
    # train_labels: n*hw*1

    t_process, t_all = time.process_time(), time.time()

    ## 
    if val_data == None and val_labels == None:
        val_data = train_data
        val_labels = train_labels

    ## == ??
    model.set_L(L)

    ## 
    sess = tf.Session(config = config,graph = model.graph)
    
    ## 
    dir_name = model.dir_name
    path = get_path(dir_name, 'summaries')
    shutil.rmtree(path, ignore_errors = True)
    writer = tf.train.SummaryWriter(path, model.graph)

    path = get_path(dir_name, 'checkpoints')
    if not os.path.isdir(path):
        os.makedirs(path)
    path = get_path(path, 'model')

    ## initialize or restore
    checkpoints = model.op_saver.last_checkpoints()
    if len(checkpoints) == 0:
        sess.run(model.op_init)
    else:
        print('{}'.format(checkpoints))
        model.op_saver.restore(sess,checkpoints[-1]) # restore weight

    ## training
    num_epochs = model.num_epochs
    batch_size = model.batch_size
    eval_frequency = model.eval_frequency

    accuracies = []
    losses =[]
    indices = collections.deque()
    num_steps = int(num_epochs * train_data.shape[0]/batch_size)
    for step in range(0, num_steps):
        if len(indices) < batch_size:
            indices.extend(np.random.permutation(train_data.shape[0]))
        idx = [indices.popleft() for ii in range(batch_size)]

        batch_data, batch_labels = train_data[idx,:], train_labels[idx]
        if type(batch_data) is not np.ndarray:
            batch_data = batch_data.toarray()

        feed_dict = {model.ph_data: batch_data, model.ph_labels: batch_labels, model.ph_dropout: model.dropout}
        learning_rate, loss_average = sess.run([model.op_train, model.op_loss_average], feed_dict)

        # periodical evaluation of the model
        if step % eval_frequency == 0 or step == num_steps:
            epoch = step * batch_size/ train_data.shape[0]
            print('step {}/{} (epoch {:.2f} / {}):'.format(step,num_steps,epoch,num_epochs)) 
            print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))       
            
            ## 
            string, accuracy, f1, loss = evaluate(model, val_data, val_labels, sess)
            accuracies.append(accuracy)
            losses.append(loss)
            print('  validation {}'.format(string))
            print('  time: {:.0f}s (all {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_all)) 

            ## summaries for tensorboard
            summary = tf.Summary()
            summary.ParseFromString(sess.run(model.op_summary, feed_dict))
            summary.value.add(tag='validation/accuracy', simple_value = accuracy)
            summary.value.add(tag='validation/f1', simple_value = f1)
            summary.value.add(tag='validation/loss', simple_value = loss)
            writer.add_summary(summary, step)

            #
            model.op_saver.save(sess, path, global_step = step)

    print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
    writer.close()
    sess.close()

    t_step = (time.time()-t_all)/num_steps
    return accuracies, losses, t_step



def evaluate(model, data, labels, sess = None):

    t_process, t_all = time.process_time(), time.time()
    predictions, loss = predict(model, data, labels, sess)
    
    ##
    ncorrects = sum(predictions == labels)
    accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average = 'weighted')
    string = 'accuracy: {:.2f} ({:d}/{:d}, f1 (weighted): {:.2f}, loss: {:.2e})'.format(accuracy, ncorrects, len(labels), f1, loss)

    if sess is None:
        string += '\n time: {:.0f}s, (all {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_all)

    return string, accuracy, f1, loss
    

def predict(model, data, labels=None, sess = None):

    loss = 0
    n = data.shape[0]
    predictions = np.empty(n)
    sess = _get_session(model,sess)
    
    batch_size = model.batch_size
    batch_data = np.zeros((batch_size, data.shape[1]))
    batch_labels = np.zeros(batch_size)
    for begin in range(0, n, batch_size):
        end = min([begin + batch_size, n])

        tmp_data = data[begin:end,:]
        if type(tmp_data) is not np.ndarray:
            tmp_data = tmp_data.toarray()
        batch_data[:end-begin] = tmp_data
        feed_dict = {model.ph_data:batch_data, model.ph_dropout: 1}

        ## compute loss
        if labels is not None:
            batch_labels[:end-begin] = labels[begin:end]
            feed_dict[model.ph_labels] = batch_labels
            batch_pred, batch_loss = sess.run([model.op_prediction, model.op_loss], feed_dict)
            loss += batch_loss
        else:
            batch_pred = sess.run([model.op_prediction], feed_dict)

        predictions[begin:end] = batch_pred[:end-begin]

    if labels is not None:
        return predictions, loss*batch_size/n
    else:
        return predictions



     
    
    


