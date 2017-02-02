import tensorflow as tf
import nn_functions as layer
import numpy as np

def network(x, choice_tots, use_dropout):

    weights = {
                #128x128
                'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1)),
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
                #64x64
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                #32x32
                'wc7': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1)),
                'wc8': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc9': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                #16x16
                'wc10': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc11': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc12': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                #8x8
                'wd1': tf.Variable(tf.truncated_normal([xxx, 4096], stddev=0.1)),
                'wd2': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1)),
                'wout': tf.Variable(tf.truncated_normal([4096, choice_tots], stddev=0.1)),

    }
    biases = {
                'bc1': tf.Variable(tf.constant(0.1, shape=[64])),
                'bc2': tf.Variable(tf.constant(0.1, shape=[128])),
                'bc3': tf.Variable(tf.constant(0.1, shape=[128])),
                'bc4': tf.Variable(tf.constant(0.1, shape=[256])),
                'bc5': tf.Variable(tf.constant(0.1, shape=[256])),
                'bc6': tf.Variable(tf.constant(0.1, shape=[256])),
                'bc7': tf.Variable(tf.constant(0.1, shape=[512])),
                'bc8': tf.Variable(tf.constant(0.1, shape=[512])),
                'bc9': tf.Variable(tf.constant(0.1, shape=[512])),
                'bc10': tf.Variable(tf.constant(0.1, shape=[512])),
                'bc11': tf.Variable(tf.constant(0.1, shape=[512])),
                'bc12': tf.Variable(tf.constant(0.1, shape=[512])),
                'bd1': tf.Variable(tf.constant(0.1, shape=[4096])),
                'bd2': tf.Variable(tf.constant(0.1, shape=[4096])),
                'bout': tf.Variable(tf.constant(0.1, shape=[choice_tots]))

    }
    
    x_image = tf.expand_dims(x,3)
    conv1 = layer.conv2d(conv0,weights['wc1'],biases['bc1'])
    conv1 = tf.nn.relu(conv1)
    conv2 = layer.conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = layer.dropout(conv2,0.9,use_dropout)
    conv2 = layer.max_pool(conv2,k=2)
    conv3 = layer.conv2d(conv2,weights['wc3'],biases['bc3'])
    conv3 = tf.nn.relu(conv3)
    conv3 = layer.dropout(conv3,0.9,use_dropout)
    conv4 = layer.conv2d(conv3,weights['wc4'],biases['bc4'])
    conv4 = tf.nn.relu(conv4)
    conv4 = layer.dropout(conv4,0.8,use_dropout)
    conv4 = layer.max_pool(conv4,k=2)
    conv5 = layer.conv2d(conv4,weights['wc5'],biases['bc5'])
    conv5 = tf.nn.relu(conv5)
    conv5 = layer.dropout(conv5,0.8,use_dropout)
    conv6 = layer.conv2d(conv5,weights['wc6'],biases['bc6'])
    conv6 = tf.nn.relu(conv6)
    conv6 = layer.dropout(conv6,0.7,use_dropout)
    conv6 = layer.max_pool(conv6,k=2)
    conv7 = layer.conv2d(conv6,weights['wc7'],biases['bc7'])
    conv7 = tf.nn.relu(conv7)
    conv7 = layer.dropout(conv7,0.7,use_dropout)
    conv8 = layer.conv2d(conv7,weights['wc8'],biases['bc8'])
    conv8 = tf.nn.relu(conv8)
    conv8 = layer.dropout(conv8,0.6,use_dropout)
    conv8 = layer.max_pool(conv8,k=2)

    dense1 = tf.reshape(conv8,[-1,weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense1,weights['wd1']),biases['bd1']))
    dense1 = layer.dropout(dense1,0.5,use_dropout)
    dense2 = tf.reshape(dense1,[-1,weights['wd2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense2,weights['wd2']),biases['bd2']))
    out = tf.nn.bias_add(tf.matmul(dense2, weights['wout']),biases['bout'])
    return out, weights, biases