import pytest
import tensorflow as tf
from GAN_lib import Generator, Discriminator

@pytest.fixture
def generator():
    return Generator()

@pytest.fixture
def discriminator():
    return Discriminator()

def test_generator(generator):
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise)
    assert generated_image.shape == (1, 28, 28, 1)

def test_discriminator(discriminator):
    image = tf.random.normal([1, 28, 28, 1])
    decision = discriminator(image)
    assert decision.shape == (1, 1)