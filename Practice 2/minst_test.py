# importing the necessary libraries and the MNIST dataset
#import tensorflow as tf
import tensorflow.compat.v1 as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

tf.disable_v2_behavior()

#from tensorflow.examples.tutorials.mnist import input_data

mnist = tf.keras.datasets.mnist.load_data()