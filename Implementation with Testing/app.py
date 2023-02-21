from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image  
import PIL  
import ArtGAN

    
   

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate():
    # call your GAN model and generate an image
    # Load the trained GAN model
    model = Generator()
    model.load_weights('./Models/generator_weights') 
    noise=tf.random.normal([1, 100])
    image=(model(noise))
    fake_image_array = np.array(image[0])
    
    # Create PIL Image object from NumPy array
    fake_image_pil = Image.fromarray(fake_image_array,'RGB')
    # save the image to a file
    image_path = './static/images/generated_image.png'  
    # Save PIL Image object to file
    fake_image_pil.save(image_path)    

    # render the HTML page with the image tag
    return render_template('generated_image.html', image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)

