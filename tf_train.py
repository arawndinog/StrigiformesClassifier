import tensorflow as tf
import numpy as np
import random
from time import time

import config_test as config
import networks.vggnet as ann

def main():
    session_id = config.session_id
    ckpt_dir = config.ckpt_dir
    session_dir = config.session_dir
    train_f = config.train_f
    valid_f = config.valid_f
    ckpt_name = config.ckpt_name

    use_ckpt = config.use_ckpt
    if use_ckpt:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epoch = config.epoch

    # train
    train_data, train_label = retrieve.extract_hdf5(train_f)

    # validation
    valid_data, valid_label = retrieve.extract_hdf5(valid_f)
    valid_data = preprocess.gen_data_feed(valid_data)

    train_data_len = len(train_data)

    choice_tots = 1000
    random.seed(1)

    startTime = time()

    x = tf.placeholder(tf.float32, [None, 128,128])
    y = tf.placeholder(tf.int64, [None])

    y_conv, neurons = ann.network(x, choice_tots, True)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    prediction = tf.argmax(y_conv,1)
    correct_prediction = tf.equal(prediction,y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=40)

    sess.run(tf.initialize_all_variables())
    if use_ckpt and ckpt:
        ckpt_path = ckpt.model_checkpoint_path
        saver.restore(sess,ckpt_path)

    global_iter = 0
    max_acc = 0
    min_err = 0
    for epoch_i in xrange(epoch):
        start_i = 0

        # combined = shuffle_list(list(zip(train_data, train_label)))
        # shuffled_train_data, shuffled_train_label = zip(*combined)

        # shuffled_train_data = preprocess.gen_data_feed(shuffled_train_data)

        while start_i < train_data_len:
            end_i = start_i + batch_size if (start_i + batch_size) < train_data_len else train_data_len
            mini_batch_datum = shuffled_train_data[start_i:end_i]
            mini_batch_label = shuffled_train_label[start_i:end_i]
            sess.run(optimizer, feed_dict = {x:mini_batch_datum, y:mini_batch_label})
            start_i += batch_size
            if global_iter%500 == 0:
                acc_stat = sess.run(accuracy, feed_dict = {x:valid_data, y:valid_label})
                err_stat = sess.run(cost, feed_dict = {x:mini_batch_datum, y:mini_batch_label})

                if acc_stat > max_acc or err_stat < min_err:
                    save_tensors(session_dir, sess, weights, biases, epoch_i, global_iter)
                    saver.save(sess, ckpt_dir + ckpt_name, global_step=global_iter)
                    max_acc = acc_stat
                    min_err = err_stat
                    print "New max accuracy reached, iteration %d checkpoint saved" % global_iter

            global_iter += 1

    print "Total time elapsed: %.3f seconds" % (time() - startTime)
    
if __name__ == '__main__':
    main()