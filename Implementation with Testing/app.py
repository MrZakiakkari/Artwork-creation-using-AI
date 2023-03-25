from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import time
from skimage.io import imsave, imread
from GAN_lib import Generator

   

app = Flask(__name__)




@app.route('/')
def index():
        # call the pretrained generator model and generate an image
        model = Generator()
        model.load_weights('./Models/generator_weights') 
        noise=tf.random.normal([1, 100])
        image=(model(noise))
    
        
        #specify path and name, the name should have random number to deal with flask caching problem
        new_image_name = "./static/images/image_" + str(time.time_ns()) + ".png"
        
        #delete previously generated images in the directory
        for filename in os.listdir('static/images/'):
            
            os.remove('static/images/' + filename)
            
        #save the image to the specified directory
        imsave(new_image_name,np.array(image[0])) 
        
        # render the HTML page with the generated image
        return render_template('generated_image.html' ,image_url=new_image_name)

@app.route('/generate-image', methods=['POST','GET'])
def generate():
    # get the values from the HTML form

    if request.method == "POST": #if generate image button was pressed
        # call the pretrained generator model and generate an image
        model = Generator()
        model.load_weights('./Models/generator_weights') 
        noise=tf.random.normal([1, 100])
        image=(model(noise))
    
        
        #specify path and name, the name should have random number to deal with flask caching problem
        new_image_name = "./static/images/image_" + str(time.time_ns()) + ".png"
        
        #delete previously generated images in the directory
        for filename in os.listdir('static/images/'):
            
            os.remove('static/images/' + filename)
            
        #save the image to the specified directory
        imsave(new_image_name,np.array(image[0])) 
        
        # render the HTML page with the generated image
        return render_template('generated_image.html' ,image_url=new_image_name)
    else: #save image button was pressed
        #get the name of the recent generated image
        for filename in os.listdir('static/images/'):
            new_image_name='./static/images/'+filename
        #copy and paste this image to the saved image directory
        imsave('./static/saved_images/'+filename,imread('./static/images/'+filename)) 
        # render the HTML page with the generated image
        return render_template('generated_image.html' ,image_url=new_image_name)        




if __name__ == '__main__':
    app.run()

