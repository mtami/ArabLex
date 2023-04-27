from tqdm import tqdm
from utils import clean_str
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine as cosine_distance


class WordSimilarity:
    def __init__(self, model_name="asafaya/bert-base-arabic"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.vocab = list(self.tokenizer.get_vocab().keys())
        self.word_top_similar = {}
        self.word_embeddings = {}
        # for word in tqdm(self.vocab):
        #     self.word_embeddings[word] = self.get_word_embedding(word)

    def get_word_embedding(self, word):
        word = clean_str(word)
        if word in self.word_embeddings:
            return self.word_embeddings[word]

        input_ids = self.tokenizer.encode(word, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(input_ids)[0][0, 1:-1].mean(dim=0)
        return embeddings

    def cosine_distance(self, word1, word2):
        word1_embedding = self.get_word_embedding(word1)
        word2_embedding = self.get_word_embedding(word2)
        return cosine_distance(word1_embedding, word2_embedding)

    def cosine_similarity(self, word1, word2):
        return 1 - self.cosine_distance(word1, word2)

    def _build_word_similaritites(self, word):
        if word in self.word_top_similar:
            return
        word_embedding = self.get_word_embedding(word)
        similarities = []
        for candidate in tqdm(self.vocab):
            candidate_embedding = self.get_word_embedding(candidate)
            similarity = 1 - cosine_distance(word_embedding, candidate_embedding)
            similarities.append((candidate, similarity))

        self.word_top_similar[word] = similarities

    def top_k(self, word, k=10):
        self._build_word_similaritites(word)
        top_similar_words = sorted(self.word_top_similar[word], key=lambda x: x[1], reverse=True)[:k]
        return top_similar_words

