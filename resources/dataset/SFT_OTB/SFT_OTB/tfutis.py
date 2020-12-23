'''
utis.py
'''

import numpy as np
import tensorflow as tf
from utis import get_path


'''
_get_session(model, sess = None)
'''


def _get_session(model, sess = None):
    if sess is None:
        sess = tf.Session(graph = model.graph)
        path = get_path(model.dir_name, 'checkpoints')
        filename = tf.train.latest_checkpoint(path)
        model.op_saver.restore(sess, filename)
    return sess
