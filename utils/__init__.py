from PIL import Image

import tensorflow as tf

import numpy as np
import math


def convolution(input_, shape, strides, padding, init_w, init_b, name, index):

    weight = tf.get_variable(
        '%s_weight_%d' % (name, index), shape=shape, initializer=init_w, dtype=tf.float32
    )
    bias = tf.get_variable(
        '%s_bias_%d' % (name, index), shape=shape[-1], initializer=init_b, dtype=tf.float32
    )

    conv = tf.nn.conv2d(
        input_, weight, strides, padding, name='%s_conv2d_%d' % (name, index)
    ) + bias

    return conv, weight, bias


def shave(image, scale):
    return image[scale: -scale, scale: -scale]


def psnr(gt, sr):

    diff = gt - sr
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2))

    return 20 * math.log10(1. / rmse)


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def save_image(image, path):

    image = image * 255
    image = image.clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode='L').convert('RGB').save(path)
