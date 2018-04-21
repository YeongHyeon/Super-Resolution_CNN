import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.developer as developer
developer.print_stamp()

def main():
    print("")
    # dataset = dman.DataSet()
    srnet = nn.SRNET()

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=100, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
