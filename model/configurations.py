
import tensorflow as tf


config = tf.app.flags


config.DEFINE_bool('is_train', False, 'set True for train')

# model name
config.DEFINE_string('model_name', 'vdsr', 'model name')
config.DEFINE_string('checkpoint_path', 'checkpoint', 'checkpoint directory')

# if you want to run this implementation on CPU, set the second parameter as '/cpu:0'
config.DEFINE_string('device', '/gpu:0', 'device for operation')

# training configuration

config.DEFINE_string('data_path', './data/train_data/vdsr_train.h5', 'training dataset path')

config.DEFINE_integer('epochs', 80, 'maximum epochs for training')
config.DEFINE_integer('batch_size', 64, 'number of datas in mini batch')
config.DEFINE_float('learning_rate', 1e-1, 'learning rate for optimization')
config.DEFINE_float('momentum_rate', 9e-1, 'momentum')

# regularization parameter
config.DEFINE_float('reg_parameter', 1e-4, 'regularization parameter')

# - learning rate decay parameters
config.DEFINE_integer('decay_step', 20, 'learning rate for optimization')
config.DEFINE_float('decay_rate', 1e-1, 'learning rate for optimization')

# adjustable gradient clipping parameter (theta)
config.DEFINE_float('grad_clip', 1e-2, 'adjustable gradient clipping parameter')

# testing configuration
config.DEFINE_integer('scale', 2, 'low resolution scale')
