from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import errno

import custom_input_storm_twocolor
import input_data_storm_twocolor
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None


def generate_metadata_file(labels,size):

    def save_metadata(file):
        with open(file, 'w') as f:
            for i in range(size):
                c = np.nonzero(labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

    # save metadata file
    save_metadata('./' + '/projector/metadata.tsv')

def deepnn(x):
    x_raw = x
    imagesize = 13

    with tf.name_scope('fc'):
        W_fc = weight_variable([imagesize * imagesize, 4096])
        b_fc = bias_variable([4096])

        h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)

    with tf.name_scope('fc0'):
        W_fc0 = weight_variable([4096, 2048])
        b_fc0 = bias_variable([2048])

        h_fc0 = tf.nn.relu(tf.matmul(h_fc, W_fc0) + b_fc0)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([2048, 1024])
        b_fc1 = bias_variable([1024])

        h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, W_fc0, W_fc1, W_fc2, keep_prob, x_raw


def batch_norm(x, n_out, phase_train):

    with tf.name_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                           name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_1x1 downsamples a feature map by 1X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x1(x):
    """max_pool_1x1 downsamples a feature map by 1X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main(_):
    # Import data

    mnist = input_data_storm_twocolor.read_data_sets('', one_hot=True)

    sess = tf.Session()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 13*13])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Build the graph for the deep net
    y_conv, W_fc0, W_fc1, W_fc2, keep_prob, input_raw = deepnn(x)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        starter_learning_rate = 1e-5
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.1, staircase=True)
        tf.summary.scalar('learning rate', learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.cpt_file)

    DATA_DIR = '../evaluation_data/fig4_color/'
    OUT_DIR = '../evaluation_data/'

    dataindex = [6, 7, 8]

    for i in range(len(dataindex)):
        TEST_IMAGES = DATA_DIR + 'test_data_ref_' + str(dataindex[i]) + '.mat'
        TEST_LABELS = DATA_DIR + 'test_label_ref_' + str(dataindex[i]) + '.mat'
        TEST_COORDINATES = DATA_DIR + 'test_coordinate_ref_' + str(dataindex[i]) + '.mat'

        test_images_n = custom_input_storm_twocolor.extract_images_from_mat(TEST_IMAGES, var_name='var')
        test_labels_n = custom_input_storm_twocolor.extract_labels_from_mat(TEST_LABELS, var_name='var')
        test_coordinates_n = custom_input_storm_twocolor.extract_labels_from_mat(TEST_COORDINATES, var_name='var')

        for k in range(np.shape(test_images_n)[1]):
           start=time.time()
           yconv = sess.run(y_conv, feed_dict={
               x: test_images_n[0,k], y_: test_labels_n[0,k], keep_prob: 1.0
               })
           yref = test_labels_n[0,k]
           end=time.time()

           corindex_a = yconv[:,0] > yconv[:,1]
           corindex_c = yconv[:,0] < yconv[:,1]

           print('af647/cf568 %d-%d' % (np.sum(corindex_a),np.sum(corindex_c)))

           if (k == 0 and i == 0):
               print('%d-%d' % (k,i))
               # print('first loop')
               ylbl = yref
               ycoor = test_coordinates_n[0,k]
               yconv_t = yconv
               yconv_a = yconv[corindex_a,:]
               yconv_c = yconv[corindex_c,:]
               # yimg = test_images_n[0,k]
           else:
               ylbl = np.append(ylbl, yref, axis=0)
               ycoor = np.append(ycoor, test_coordinates_n[0,k], axis=0)
               yconv_t = np.append(yconv_t, yconv, axis=0)
               yconv_a = np.append(yconv_a, yconv[corindex_a,:], axis=0)
               yconv_c = np.append(yconv_c, yconv[corindex_c,:], axis=0)
               # yimg = np.append(yimg, test_images_n[0,k], axis=0)

    savefilename = DATA_DIR + FLAGS.result_name

    if not os.path.exists(os.path.dirname(savefilename)):
        try:
            os.makedirs(os.path.dirname(savefilename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    sio.savemat(savefilename, mdict={'yconv_a':yconv_a, 'yconv_c':yconv_c, 'yconv_t': yconv_t, 'ycoor':ycoor, 'ylbl':ylbl})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_file', type=str,
                        default='../trained_models/ref_storm_color/result.ckpt',
                        help='Input checkpoint file')
    parser.add_argument('--result_name', type=str,
                        default='nn_results_color.mat',
                        help='Test results')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
