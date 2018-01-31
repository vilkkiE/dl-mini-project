from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile

from read_data import get_data
from itertools import product

import tensorflow as tf


def deepnn(x, phase_train):
    """deepnn builds the graph for a deep net for classifying objects.
    Args:
      x: an input tensor with the dimensions (N_examples, 1024*3), where 1024 is the
      number of pixels and 3 the number of channels in a standard CIFAR10 image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes.
      keep_prob is a scalar placeholder for the probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is 3 for RGB images
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 32, 32, 3])
        # Pre-process
        x_image = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x_image)

    # First convolutional layer - maps one image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable_conv2d('W_conv1', [5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        conv1_output = conv2d(x_image, W_conv1) + b_conv1
        h_conv1 = tf.nn.relu(conv1_output)
        conv1_bn = batch_norm(h_conv1, 32, phase_train)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(conv1_bn)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable_conv2d('W_conv2', [5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        conv2_output = conv2d(h_pool1, W_conv2) + b_conv2
        h_conv2 = tf.nn.relu(conv2_output)
        conv2_bn = batch_norm(h_conv2, 64, phase_train)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(conv2_bn)

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 8x8x64 feature maps -- maps this to 384 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable('W_fc1', [8 * 8 * 64, 384])
        b_fc1 = bias_variable([384])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully connected layer 2 -- after 2 round of downsampling, our 32x32 image
    # is down to 384 features -- maps this to 192 features.
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable('W_fc2', [384, 192])
        b_fc2 = bias_variable([192])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout2'):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable('W_fc3', [192, 10])
        b_fc3 = bias_variable([10])

        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)


def weight_variable_conv2d(name, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
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


def main(_):
    # Import data
    train_data, train_labels, test_data, test_labels, meta_data = get_data()
    tf_learning_rate = tf.placeholder(tf.float32, name='tf_learning_rate')
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3072])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x, phase_train)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(tf_learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())

    learning_rates = [0.01, 0.0005, 0.0001, 0.00005, 0.00001]
    batch_sizes = [50, 100]
    epochs = 20
    for learning_rate, batch_size in product(learning_rates, batch_sizes):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train' +'/batch_size-' + str(batch_size) + "_learning_rate-" + str(learning_rate), sess.graph)
            test_writer = tf.summary.FileWriter("./logs/nn_logs" + '/test' + '/batch_size-' + str(batch_size) + "_learning_rate-" + str(learning_rate),sess.graph)
            print('learning rate: %g, batch size: %d' % (learning_rate, batch_size))
            batches = int(50000 / batch_size)
            for k in range(epochs):
                for i in range(batches):
                    batch_start = i*batch_size
                    batch = train_data[batch_start:batch_start+batch_size]
                    batch_labels = train_labels[batch_start:batch_start+batch_size]
                    if i == batches - 1:
                        summary, train_acc = sess.run([merged, accuracy],feed_dict={
                            x: batch, y_: batch_labels, keep_prob: 1.0, phase_train: False, tf_learning_rate: learning_rate})
                        print('Epoch %d, training accuracy %g' % (k, train_acc))
                        train_writer.add_summary(summary, k)
                    train_step.run(feed_dict={x: batch, y_: batch_labels, keep_prob: 0.5, phase_train: True, tf_learning_rate: learning_rate})
                summary, test_acc = sess.run([merged, accuracy], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0, phase_train: False, tf_learning_rate: learning_rate})
                test_writer.add_summary(summary, k)
            print('test accuracy %g \n' % accuracy.eval(feed_dict={
                x: test_data, y_: test_labels, keep_prob: 1.0, phase_train: False, tf_learning_rate: learning_rate}))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])