import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set GPU as the device to ensure that we are not running on the cpu
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Num GPUs Logical: ", len(logical_gpus))




class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.reshape = tf.keras.layers.Reshape((8, 8, 256))
        self.convT2 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.convT3 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()   
        self.convT4 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()         
        self.convT5 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.convT2(x)
        x = self.batchnorm2(x, training=training)
        x = self.relu(x)
        x = self.convT3(x)
        x = self.batchnorm3(x, training=training)
        x = self.relu(x)    
        x = self.convT4(x)
        x = self.batchnorm4(x, training=training)
        x = self.relu(x)         
        x = self.convT5(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.batchnorm3 = tf.keras.layers.BatchNormalization()     
        self.conv4 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.batchnorm4 = tf.keras.layers.BatchNormalization()        
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        
        x = self.batchnorm4(x, training=training)
        x = self.leaky_relu(x)        
        x = self.flatten(x)
        x = self.fc1(x)
        return x



class ArtGAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,
                                                                    beta_1=0.5,
                                                                    beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,
                                                                    beta_1=0.5,
                                                                    beta_2=0.999)
        filenames = tf.data.Dataset.list_files("C:/Users/alakk/Github/New Final Year Project/resized/*.jpg")
        
        self.base_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg',
                                                weights='imagenet')
        self.dataset = filenames.map(self.load_and_preprocess_image)  
        self.batch_size=128

        self.dataset = self.dataset.batch(self.batch_size)
        
        
    def load_and_preprocess_image(self,filename): 
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [128, 128])
        image = (image/255.0-0.5)*2  # normalize to [-1,1] range to be consistent with the generator
        return image
    
    
    
    def generate_images(self):
        noise=tf.random.normal([25, 100])
        images=self.generator(noise)
        images=images/2+0.5
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i],cmap=plt.cm.binary)
        plt.show()
    
    def mode_collapse_detection(self,generate_images):
        mean = tf.reduce_mean(generate_images, axis=0)
        distance = tf.reduce_mean(tf.norm(generate_images - mean, axis=-1))
        return distance


    # Compute activations for real images
    def compute_activations(self,images, batch_size=32):
        activations = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_images = tf.image.resize(batch_images, (299, 299))
            activations.append(self.base_model(batch_images).numpy())
        return np.concatenate(activations, axis=0)

    def fid_score(self,real_images, generated_images):
        # Compute activations for real and generated images
        real_activations = self.compute_activations(real_images)
        generated_activations = self.compute_activations(generated_images)
        # Compute mean and covariance for real and generated activations
        real_mean = np.mean(real_activations, axis=0)
        real_cov = np.cov(real_activations, rowvar=False)
        generated_mean = np.mean(generated_activations, axis=0)
        generated_cov = np.cov(generated_activations, rowvar=False)

        # Compute Fréchet distance
        diff = real_mean - generated_mean
        fid = np.sum(diff**2) + np.trace(real_cov) + np.trace(generated_cov) - 2 * np.trace(scipy.linalg.sqrtm(np.dot(real_cov, generated_cov)))
        return fid
    
    def train(self, epochs):
        self.FID_scores=[]
        
        self.mode_collapse_scores=[]
        for epoch in range(epochs):
            for images in self.dataset:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    noise = tf.random.normal([self.batch_size, 100])
                    generated_images = self.generator(noise, training=True)
                    #mode_collapse_distance = mode_collapse_detection(generated_images)

                    real_output = self.discriminator(images, training=True)
                    fake_output = self.discriminator(generated_images, training=True)

                    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output)))
                    disc_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))) +
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))))

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                
                print("Epoch {}/{}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(epoch+1, epochs, gen_loss.numpy(), disc_loss.numpy()))
            self.generate_images()
            fid = self.fid_score(images/2.0+0.5, generated_images/2.0+0.5)
            self.FID_scores.append(fid)
            
            mode_collapse_distance = self.mode_collapse_detection(generated_images/2.0+0.5)
            self.mode_collapse_scores.append(mode_collapse_distance)
            plt.figure(figsize=(8, 4))

            plt.plot(self.FID_scores)
            plt.xlabel("epcoch", fontsize=14)
            plt.ylabel("FID scores", fontsize=14)

            plt.show()
            
            plt.figure(figsize=(8, 4))

            plt.plot(self.mode_collapse_scores)
            plt.xlabel("epcoch", fontsize=14)
            plt.ylabel("mode collapse scores", fontsize=14)

            plt.show()            
            print("FID score: {:.4f}, Mode collapse: {:.4f}".format(fid,mode_collapse_distance))
