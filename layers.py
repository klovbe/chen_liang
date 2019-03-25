#encoding=utf-8
import tensorflow as tf


def batch_norm(x, is_training, epsilon=1e-3, momentum=0.99, name=None):
    """Code modification of http://stackoverflow.com/a/33950177"""
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                        center=True, scale=True, is_training=is_training, scope=name)
