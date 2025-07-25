Brand Personality Analysis Tool
This Python program analyzes images associated with a brand to uncover subconscious perceptions of its personality, based on Jennifer Aaker's brand personality framework. It uses deep learning and natural language processing to process images, extract labels, and map them to personality traits.
Features

Image Input: Accepts Google Drive links to images (JPG format) representing thoughts and feelings about a brand.
Image Processing: Uses EfficientNet V2 (via TensorFlow Hub) to classify images and extract labels.
Label Expansion: Expands image labels using WordNet synonyms and Google News word vectors for semantic similarity.
Personality Mapping: Maps labels to Aaker's brand personality dimensions and primary personality traits.
Visualization: Generates a word cloud of personality traits and a bar chart of Aaker's personality dimensions.

Prerequisites

Python 3.7+
Google Drive account (for uploading images)
Internet connection (for downloading models and word vectors)
Libraries: tensorflow, tensorflow_hub, gensim, nltk, PIL, matplotlib, numpy, regex, gdown, wordcloud

Installation

Clone or download this repository.
Install required packages:pip install tensorflow tensorflow-hub gensim nltk pillow matplotlib numpy regex gdown wordcloud


Download NLTK resources:import nltk
nltk.download('stopwords')
nltk.download('wordnet')



Usage

Prepare Images:

Select 8–10 images (JPG format) representing your thoughts and feelings about a brand, avoiding direct brand imagery.
Upload images to Google Drive and set sharing to "Anyone with the link."
Copy the shareable links for each image.


Run the Program:

Execute the script in a Python environment (e.g., Jupyter Notebook or Colab).
When prompted, paste Google Drive image links one by one. Enter N or NO to finish.
Example input:Please paste the Google Drive link of your first image.
Make sure you give the access permission to the images link: https://drive.google.com/file/d/abc123/view?usp=sharing
Please paste the Google Drive link of your next image (otherwise write N or NO): https://drive.google.com/file/d/xyz456/view?usp=sharing
...




Output:

A subplot displaying uploaded images.
A word cloud of primary personality traits.
A bar chart of Aaker's brand personality dimensions with similarity scores.
Console output showing the trait similarity rate and number of traits identified.



Example
For testing, use sample images from an in-depth interview about the Red Bull brand:Red Bull Sample Images
Configuration
Adjust parameters in the main_function for fine-tuning:

w_similarity_rate: Word similarity threshold (default: 0.5).
tr_similarity_rate: Trait similarity threshold (default: 0.5).
ak_similarity_rate: Aaker personality similarity threshold (default: 0.3).
min_of_associations: Minimum number of personality traits (default: 5).
max_of_associations: Maximum number of personality traits (default: 10).
tuning_tolerance: Step size for adjusting similarity thresholds (default: 0.01).

Example:
main_function(w_similarity_rate=0.5, tr_similarity_rate=0.5, ak_similarity_rate=0.3, min_of_associations=5, max_of_associations=10, tuning_tolerance=0.01)

How It Works

Image Input: Collects Google Drive links via img_links_input().
Image Processing: Downloads and resizes images to 224x224, converting them to arrays.
Label Prediction: Uses EfficientNet V2 to predict ImageNet labels.
Label Expansion: Expands labels with synonyms (WordNet) and similar words (Google News vectors).
Personality Mapping:
Maps expanded labels to primary personality traits and Aaker's dimensions.
Adjusts similarity thresholds to ensure 5–10 traits are identified.


Visualization:
Displays images in a grid.
Generates a word cloud of traits.
Plots Aaker's personality dimensions as a bar chart.



Data Sources

ImageNet Labels: Pre-trained EfficientNet V2 model for image classification.
Google News Word Vectors: Pre-trained word embeddings for semantic similarity.
Aaker's Brand Personality: CSV file with personality dimensions and traits.
Common Words: Lists of adjectives, verbs, and nouns relevant to brand personality.
Primary Personality Traits: List of human personality traits for mapping.

Notes

Ensure Google Drive links are accessible ("Anyone with the link").
Images must be in JPG format.
The program dynamically adjusts similarity thresholds to balance the number of traits.
For Colab users, files are downloaded to /content/.

Limitations

Requires internet access for downloading models and word vectors.
Image classification accuracy depends on the EfficientNet V2 model.
Results may vary based on image quality and relevance to brand perceptions.

License
This project is licensed under the MIT License.
