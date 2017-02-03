import tensorflow as tf
import nn_functions as layer

def network(x, choice_tots, use_dropout):

    neurons = {
                #128x128
                'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1)),
                'bc1': tf.Variable(tf.constant(0.1, shape=[64])),
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
                'bc2': tf.Variable(tf.constant(0.1, shape=[128])),
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
                'bc3': tf.Variable(tf.constant(0.1, shape=[128])),
                #64x64
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'bc4': tf.Variable(tf.constant(0.1, shape=[256])),
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'bc5': tf.Variable(tf.constant(0.1, shape=[256])),
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'bc6': tf.Variable(tf.constant(0.1, shape=[256])),
                #32x32
                'wc7': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1)),
                'bc7': tf.Variable(tf.constant(0.1, shape=[512])),
                'wc8': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'bc8': tf.Variable(tf.constant(0.1, shape=[512])),
                'wc9': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'bc9': tf.Variable(tf.constant(0.1, shape=[512])),
                #16x16
                'wc10': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'bc10': tf.Variable(tf.constant(0.1, shape=[512])),
                'wc11': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'bc11': tf.Variable(tf.constant(0.1, shape=[512])),
                'wc12': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'bc12': tf.Variable(tf.constant(0.1, shape=[512])),
                #8x8
                'wd1': tf.Variable(tf.truncated_normal([xxx, 4096], stddev=0.1)),
                'bd1': tf.Variable(tf.constant(0.1, shape=[4096])),
                'wd2': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1)),
                'bd2': tf.Variable(tf.constant(0.1, shape=[4096])),
                'wout': tf.Variable(tf.truncated_normal([4096, choice_tots], stddev=0.1)),
                'bout': tf.Variable(tf.constant(0.1, shape=[choice_tots]))
    }
    
    x_image = tf.expand_dims(x,3)
    #128x128
    conv1 = layer.conv2d(x_image,neurons['wc1'],neurons['bc1'])
    conv1 = tf.nn.relu(conv1)
    conv2 = layer.conv2d(conv1,neurons['wc2'],neurons['bc2'])
    conv2 = tf.nn.relu(conv2)
    conv3 = layer.conv2d(conv2,neurons['wc3'],neurons['bc3'])
    conv3 = tf.nn.relu(conv3)
    conv3 = layer.max_pool(conv3,k=2)
    #64x64
    conv4 = layer.conv2d(conv3,neurons['wc4'],neurons['bc4'])
    conv4 = tf.nn.relu(conv4)
    conv5 = layer.conv2d(conv4,neurons['wc5'],neurons['bc5'])
    conv5 = tf.nn.relu(conv5)
    conv6 = layer.conv2d(conv5,neurons['wc6'],neurons['bc6'])
    conv6 = tf.nn.relu(conv6)
    conv6 = layer.max_pool(conv6,k=2)
    #32x32
    conv7 = layer.conv2d(conv6,neurons['wc7'],neurons['bc7'])
    conv7 = tf.nn.relu(conv7)
    conv8 = layer.conv2d(conv7,neurons['wc8'],neurons['bc8'])
    conv8 = tf.nn.relu(conv8)
    conv9 = layer.conv2d(conv8,neurons['wc9'],neurons['bc9'])
    conv9 = tf.nn.relu(conv9)
    conv9 = layer.max_pool(conv9,k=2)
    #16x16
    conv10 = layer.conv2d(conv9,neurons['wc10'],neurons['bc10'])
    conv10 = tf.nn.relu(conv10)
    conv11 = layer.conv2d(conv10,neurons['wc11'],neurons['bc11'])
    conv11 = tf.nn.relu(conv11)
    conv12 = layer.conv2d(conv11,neurons['wc12'],neurons['bc12'])
    conv12 = tf.nn.relu(conv12)
    conv12 = layer.max_pool(conv12,k=2)
    #8x8
    dense1 = tf.reshape(conv12,[-1,neurons['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense1,neurons['wd1']),neurons['bd1']))
    dense1 = layer.dropout(dense1,0.5,use_dropout)
    dense2 = tf.reshape(dense1,[-1,neurons['wd2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense2,neurons['wd2']),neurons['bd2']))
    dense2 = layer.dropout(dense2,0.5,use_dropout)
    out = tf.nn.bias_add(tf.matmul(dense2, neurons['wout']),neurons['bout'])
    return out, neurons