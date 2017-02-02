import tensorflow as tf

def conv2d(img, W, b, padding_type = 'SAME'):
    return tf.nn.bias_add(tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding=padding_type),b)

def max_pool(img, k=2, s=2):
    return tf.nn.max_pool(img,ksize=[1,k,k,1], strides=[1,s,s,1], padding='SAME')

def avg_pool(img, k=2, s=2):
    return tf.nn.avg_pool(img,ksize=[1,k,k,1], strides=[1,s,s,1], padding='SAME')

def dropout(img, keep_prob, backprop):
    if backprop:
        return tf.nn.dropout(img,keep_prob)
    return img

def relu_leaky(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x