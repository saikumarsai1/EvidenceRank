import json
from preprocessing import preprocess


with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


for paper in papers:
    text = paper["title"] + " " + paper["abstract"]
    tokens = preprocess(text)

    print("\n", paper["id"])
    print(tokens)