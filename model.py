from utils import clean_str
import requests
import gzip
import shutil
import streamlit as st
import gensim


@st.experimental_memo
def download_fastext():
    # Download the Arabic FastText model
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz"
    response = requests.get(url, stream=True)
    with open("cc.ar.300.bin.gz", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the .bin file from the .gz archive
    with gzip.open("cc.ar.300.bin.gz", "rb") as f_in:
        with open("cc.ar.300.bin", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


model = gensim.models.fasttext.load_facebook_vectors('./cc.ar.300.bin')


def get_top_k(word, k=10):
    cleaned_str = clean_str(word)
    return model.most_similar(cleaned_str, topn=20)


def get_similarity(w1, w2):
    return model.similarity(clean_str(w1), clean_str(w2))
