import string


def preprocess_text(text: str) -> str:
    mytable = str.maketrans("", "", string.punctuation)
    return text.lower().translate(mytable)
