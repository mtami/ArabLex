import datetime

from model import WordSimilarity
from utils import load_words

start_date = datetime.date(2023, 2, 1)


WORDS = [word for word in load_words()]

word_sim = WordSimilarity()


def calculate_similarity(day: int, word: str) -> tuple[bool, dict]:
    today = datetime.date.today()
    word_index = (today - start_date).days
    if day not in range(0, word_index + 1):
        return False, {"detail": "Bad day!"}

    similarity = word_sim.cosine_similarity(word, WORDS[day])

    return True, {"word": word, "similarity": float(similarity)}
