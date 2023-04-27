import datetime

from model import WordSimilarity
from utils import load_words, scale_val

start_date = datetime.date(2023, 2, 1)


WORDS = [word for word in load_words()]

word_sim = WordSimilarity()


def calculate_distance(day: int, word: str) -> tuple[bool, dict]:
    today = datetime.date.today()
    word_index = (today - start_date).days
    if day not in range(0, word_index + 1):
        return False, {"detail": "Bad day!"}

    distance = word_sim.cosine_distance(word, WORDS[day])
    scaled_distance = scale_val(distance)

    return True, {"word": word, "distance": int(scaled_distance)}
