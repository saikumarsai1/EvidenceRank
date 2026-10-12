import json
from rank_bm25 import BM25Okapi

from preprocessing import preprocess


with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


documents = []

for paper in papers:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    text = title + " " + abstract

    tokens = preprocess(text)

    # BM25 needs a LIST of words, not one string
    if isinstance(tokens, str):
        tokens = tokens.split()

    documents.append(tokens)


bm25 = BM25Okapi(documents)


def search(query, top_k=3):

    query_tokens = preprocess(query)

    # BM25 needs tokenized query
    if isinstance(query_tokens, str):
        query_tokens = query_tokens.split()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    for index in ranked_indices:
        index = int(index)

        print(
            papers[index].get("id", "N/A"),
            "-",
            papers[index].get("title", "No title"),
            "| Score:",
            round(float(scores[index]), 3)
        )


query = input("Enter your search query: ")

print("\nBM25 results:")
search(query)