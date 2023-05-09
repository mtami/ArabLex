import datetime

from model import WordSimilarity, get_similarity, get_top_k
from utils import load_words

start_date = datetime.date(2023, 2, 1)


SECRET_WORDS = [word for word in load_words()]
SECRET_WORDS_TOP_K = {}


def is_valid_day(day):
    today = datetime.date.today()
    word_index = (today - start_date).days
    if day not in range(0, word_index + 1):
        return False
    return True


def calculate_similarity(day: int, word: str) -> tuple[bool, dict]:
    if not is_valid_day(day):
        return False, {"detail": "Bad day!"}

    similarity = get_similarity(word, SECRET_WORDS[day])

    return True, {"word": word, "similarity": float(similarity)}


def suggest_similar_word(day: int):
    if not is_valid_day(day):
        return False
    secret_word = SECRET_WORDS[day]
    if secret_word not in SECRET_WORDS_TOP_K.keys():
        top_k = get_top_k(secret_word)
        SECRET_WORDS_TOP_K[secret_word] = top_k
    else:
        top_k = SECRET_WORDS_TOP_K[secret_word]

