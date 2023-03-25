# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:28:56 2023

@author: alakk
"""

import tensorflow as tf
from GAN_lib import Generator, Discriminator, ArtGAN
import numpy as np
from PIL import Image  
import PIL  
import matplotlib.pyplot as plt
from skimage.io import imsave
if __name__ == '__main__':
    model = Generator()
    model.load_weights('./Models/generator_weights')
    noise=tf.random.normal([1, 100])
    image=(model(noise))
    plt.figure(figsize=(128,128))
    image_path = './static/images/generated_image.png'
    imsave(image_path,np.array(image[0]))
 