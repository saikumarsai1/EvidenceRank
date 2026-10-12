import json
from collections import defaultdict
from preprocessing import preprocess


with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


inverted_index = defaultdict(list)


for paper in papers:
    text = paper["title"] + " " + paper["abstract"]
    tokens = preprocess(text)

    for token in set(tokens):
        inverted_index[token].append(paper["id"])


for term in ["air", "disease", "health", "pollution"]:
    print(term, "->", inverted_index.get(term, []))