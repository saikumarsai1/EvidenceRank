import re


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens


if __name__ == "__main__":
    text = "Air Pollution and Cardiovascular Disease!"
    print(preprocess(text))