import tensorflow as tf
import tensorflow_hub as hub
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import PIL.Image as Image
import matplotlib.pylab as plt
import numpy as np
import regex as re
import gdown
import csv
from IPython.display import clear_output

nltk.download('stopwords')
nltk.download('wordnet')


# Using the functional API to construct the model
input_tensor = tf.keras.Input(shape=(224, 224, 3))

# Wrap KerasLayer in a Lambda layer to force eager execution
output_tensor = tf.keras.layers.Lambda(lambda x: hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2",
    trainable=False  # Set trainable to False
)(x))(input_tensor)

model_img_cls = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


# url = "https://drive.google.com/uc?id=1w0nuvKlKG6OmiP1CQh8H2K4WH1l7hhF-"
# output = "GoogleNews-vectors-negative300.bin.gz"

# gdown.download(url, output, quiet=False)
