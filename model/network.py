
import tensorflow as tf

import utils


def inference(image):

    weights = []
    biases = []

    with tf.name_scope('vdsr_layer_0'):
        layer, weight, bias = utils.convolution(
            image,
            shape=[3, 3, 1, 64], strides=[1, 1, 1, 1], padding='SAME',
            init_w=tf.initializers.he_normal(),
            init_b=tf.initializers.zeros(),
            name='vdsr', index=0
        )
        layer = tf.nn.relu(layer)

    weights.append(weight)
    biases.append(bias)

    for i in range(1, 19):
        with tf.name_scope('vdsr_layer_%d' % i):
            layer, weight, bias = utils.convolution(
                layer,
                shape=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME',
                init_w=tf.initializers.he_normal(),
                init_b=tf.initializers.zeros(),
                name='vdsr', index=i
            )
            layer = tf.nn.relu(layer)

        weights.append(weight)
        biases.append(bias)

    with tf.name_scope('vdsr_layer_19'):
        layer, weight, bias = utils.convolution(
            layer,
            shape=[3, 3, 64, 1], strides=[1, 1, 1, 1], padding='SAME',
            init_w=tf.initializers.he_normal(),
            init_b=tf.initializers.zeros(),
            name='vdsr', index=19
        )
    weights.append(weight)
    biases.append(bias)

    with tf.name_scope('vdsr_output'):
        output = image + layer
        output = tf.clip_by_value(output, 0., 1.)

    return output, layer, weights, biases
