from keras.layers import Deconvolution2D, Reshape

def build_generator(start_filters, filter_size, latent_dim):
  
  # function for building a CNN block for upsampling the image
  def add_generator_block(x, filters, filter_size):
      x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x

  # input is a noise vector 
  inp = Input(shape=(latent_dim,))

  # projection of the noise vector into a tensor with 
  # same shape as last conv layer in discriminator
  x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
  x = BatchNormalization()(x)
  x = Reshape(target_shape=(4, 4, start_filters * 8))(x)

  # design the generator to upsample the image 4x
  x = add_generator_block(x, start_filters * 4, filter_size)
  x = add_generator_block(x, start_filters * 2, filter_size)
  x = add_generator_block(x, start_filters, filter_size)
  x = add_generator_block(x, start_filters, filter_size)    

  # turn the output into a 3D tensor, an image with 3 channels 
  x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
  
  return Model(inputs=inp, outputs=x)