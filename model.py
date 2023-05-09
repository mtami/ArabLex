import os
from utils import clean_str
import requests
import gzip
import shutil
import streamlit as st
import gensim

import tempfile


@st.experimental_memo
def download_fastext():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Download the Arabic FastText model
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz"
    response = requests.get(url, stream=True)
    gz_path = os.path.join(temp_dir, "cc.ar.300.bin.gz")
    with open(gz_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the .bin file from the .gz archive
    bin_path = os.path.join(temp_dir, "cc.ar.300.bin")
    with gzip.open(gz_path, "rb") as f_in:
        with open(bin_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return bin_path


@st.cache_resource
def load_fastext():
    bin_path = download_fastext()
    model = gensim.models.fasttext.load_facebook_vectors(bin_path)
    return model


model = load_fastext()


def get_top_k(word, k=10):
    cleaned_str = clean_str(word)
    return model.most_similar(cleaned_str, topn=20)


def get_similarity(w1, w2):
    return model.similarity(clean_str(w1), clean_str(w2))
