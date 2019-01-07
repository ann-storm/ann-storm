"""Neural network-based single molecule color classification.
Code structure based on the MNIST classifier tutorial by Tensorflow authors.
(https://github.com/tensorflow/tensorflow/)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector

import argparse
import sys
import tempfile
import os
import errno

import input_data_storm_twocolor
import custom_input_storm_twocolor
import tensorflow as tf
import numpy as np
import scipy.io as sio


FLAGS = None


def generate_metadata_file(labels, size):

    def save_metadata(file):
        with open(file, 'w') as f:
            for i in range(size):
                c = np.nonzero(labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

    # save metadata file
    save_metadata('./' + '/projector/metadata.tsv')


def rundircreation():
    mydir = os.path.join(
        os.getcwd(),
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    return mydir


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

    return y_conv, W_fc, W_fc0, W_fc1, W_fc2, keep_prob, x_raw


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

def conv2d_upsample(x, W, deconv_shape):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, W, strides=[1, 2, 2, 1], output_shape=deconv_shape, padding='SAME')


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
    # initial = tf.truncated_normal(shape, stddev=0.005)
    initial = tf.contrib.layers.xavier_initializer()

    return tf.Variable(initial(shape))

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    # initial = tf.constant(0.0, shape=shape)
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        mean_summary = tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        stddev_summary = tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        hist_summary = tf.summary.histogram('histogram', var)

    return mean_summary, stddev_summary, hist_summary


def main(_):
    # Import data

    rundirname = rundircreation()

    if os.path.exists("./current"):
        os.unlink("./current")
    os.symlink(rundirname, './current')

    mnist = input_data_storm_twocolor.read_data_sets('', one_hot=True)

    sess = tf.Session()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 13*13])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Build the graph for the deep net
    y_conv, W_fc, W_fc0, W_fc1, W_fc2, keep_prob, input_raw = deepnn(x)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    loss_summary = tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        starter_learning_rate = FLAGS.learn_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   60000, 0.2, staircase=True)
        # tf.summary.scalar('learning rate', learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
            
    accuracy = tf.reduce_mean(correct_prediction)

    train_summary = tf.summary.scalar('train_accuracy', accuracy)
    validation_summary = tf.summary.scalar('validation_accuracy', accuracy)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    train_writer = tf.summary.FileWriter(rundirname + '/train')

    sess.run(tf.global_variables_initializer())

    for i in range(FLAGS.num_iter):
        batch = mnist.train.next_batch(FLAGS.batch_size)
        if i % 100 == 0:
            train_accuracy_print = accuracy.eval(session=sess, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })
            validation_accuracy_print = accuracy.eval(session=sess, feed_dict={
                x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0
            })
            print('step %d, training accuracy %.5f, validation accuracy %g' % (i, train_accuracy_print, validation_accuracy_print))

        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5
        })
        tf.assign_add(global_step, 1)

        train_loss, loss_sum = sess.run(
                [cross_entropy, loss_summary], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0
                    }
                )
        train_writer.add_summary(loss_sum, i)

        train_acc, train_sum = sess.run(
                [accuracy, train_summary], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0
                    }
                )
        train_writer.add_summary(train_sum, i)

        val_acc, val_sum = sess.run(
                [accuracy, validation_summary], feed_dict={
                    x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0
                    }
                )
        train_writer.add_summary(val_sum, i)

    saver = tf.train.Saver()
    saver.save(sess, rundirname + '/result.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--num_iter', type=int,
                        default=3000,
                        help='Number of iterations')
    parser.add_argument('--learn_rate', type=float,
                        default=1e-3,
                        help='Learning rate')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
