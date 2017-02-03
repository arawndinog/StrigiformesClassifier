import tensorflow as tf
import numpy as np
import random
from time import time

import config_test as config
import networks.vggnet as ann

def main():
    x = tf.placeholder(tf.float32, [None, 128,128])
    y = tf.placeholder(tf.int64, [None])

    y_conv, weights, biases = ann.network(x, choice_tots, True)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    prediction = tf.argmax(y_conv,1)
    correct_prediction = tf.equal(prediction,y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=100)

    writer = tf.train.SummaryWriter(session_dir, sess.graph)
    sess.run(tf.initialize_all_variables())
    if use_ckpt and ckpt:
        ckpt_path = ckpt.model_checkpoint_path
        saver.restore(sess,ckpt_path)
    else:
        ckpt_path = ""
    return