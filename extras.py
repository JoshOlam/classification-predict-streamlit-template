"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from numpy import result_type
import scipy
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import sparse
import base64
import csv
import string

def clean(text):
    lemmatizer = WordNetLemmatizer()
    def word_lemma(words, lemmatizer):
        lemma = [lemmatizer.lemmatize(word) for word in words]
        return ''.join([l for l in lemma])
    def remove_extras(post):
        punc_numbers = string.punctuation + '0123456789'
        return ''.join([l for l in post if l not in punc_numbers])
    #Lower case
    text = text.lower()
    #Removal of Punctuation
    text = remove_extras(text)
    text = word_lemma(text, lemmatizer)
    return text		

