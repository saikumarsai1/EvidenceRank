import json
import os

DATA_FILE = "data/raw/papers.json"
OUTPUT_FILE = "data/raw/judgments.json"

QUERIES = [
    "air pollution cardiovascular disease",
    "air pollution respiratory disease",
    "climate change human health",
    "particulate pollution health",
    "air pollution mortality"
]

ANSWERS = [
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, False, False],
    [True, True, True, True, True],
    [True, True, True, True, True]
]

TOP_K = 5

with open(DATA_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    p.get("title", "") + " " + p.get("abstract", "")
    for p in papers
]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

vectors = vectorizer.fit_transform(documents)

judgments = {}

for query, answer_list in zip(QUERIES, ANSWERS):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        vectors
    )[0]

    ranked_indices = scores.argsort()[::-1][:TOP_K]

    judgments[query] = []

    for rank, (index, relevant) in enumerate(
        zip(ranked_indices, answer_list),
        start=1
    ):
        judgments[query].append({
            "rank": rank,
            "doc_id": papers[index]["id"],
            "relevant": relevant
        })

os.makedirs("data/raw", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        judgments,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 60)
print("JUDGMENTS CREATED SUCCESSFULLY")
print("=" * 60)
print("Queries:", len(QUERIES))
print("Judgments:", len(QUERIES) * TOP_K)
print("Saved:", OUTPUT_FILE)