import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

class SRNET(object):

    def __init__(self):

        print("\n** Initialize Super-Resolution Network")

        self.inputs = tf.placeholder(tf.float32, [None, None, None, None])
        self.outputs = tf.placeholder(tf.float32, [None, None, None, None])

        self.channel = 3
        self.ksize = 3
        self.n1 = 32
        self.n2 = 16

        self.weights = {
            'patch_ext': tf.Variable(tf.random_normal([self.ksize, self.ksize, self.channel, self.n1], stddev=np.sqrt(3))),
            'nl_map': tf.Variable(tf.random_normal([self.ksize, self.ksize, self.n1, self.n2], stddev=np.sqrt(32))),
            'recon': tf.Variable(tf.random_normal([self.ksize, self.ksize, self.n2, self.channel], stddev=np.sqrt(16))),
        }

        self.biases = {
            'patch_ext': tf.Variable(tf.random_normal([self.n1], stddev=np.sqrt(self.channel))),
            'nl_map': tf.Variable(tf.random_normal([self.n2], stddev=np.sqrt(self.n1))),
            'recon': tf.Variable(tf.random_normal([self.channel], stddev=np.sqrt(self.n2))),
        }

        self.patch_ext = tf.nn.relu(tf.add(tf.nn.conv2d(self.inputs, self.weights['patch_ext'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['patch_ext']))
        self.nonlinear_map = tf.nn.relu(tf.add(tf.nn.conv2d(self.patch_ext, self.weights['nl_map'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['nl_map']))
        self.recon = tf.add(tf.nn.conv2d(self.nonlinear_map, self.weights['recon'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['recon'])

        self.cost = tf.reduce_sum(tf.square(self.recon - self.outputs))

        self.leaning_rate = 0.001
        self.optimizer = tf.train.AdamOptimizer(self.leaning_rate).minimize(self.cost)
