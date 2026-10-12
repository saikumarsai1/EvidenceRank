import json
import math
from collections import Counter

DATA_FILE = "data/raw/papers.json"
JUDGMENT_FILE = "data/raw/judgments.json"

K = 5

with open(DATA_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

with open(JUDGMENT_FILE, "r", encoding="utf-8") as f:
    judgments = json.load(f)


# ---------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------

def tokenize(text):
    return [
        word.lower()
        for word in text.split()
        if word.isalnum()
    ]


documents = []

for paper in papers:
    text = (
        paper.get("title", "") + " " +
        paper.get("abstract", "")
    )
    documents.append(tokenize(text))


# ---------------------------------------------------------
# TF-IDF BASELINE
# ---------------------------------------------------------

N = len(documents)

df = Counter()

for doc in documents:
    for term in set(doc):
        df[term] += 1


def tfidf_vector(doc):
    counts = Counter(doc)
    vector = {}

    for term, count in counts.items():
        tf = 1 + math.log(count)
        idf = math.log((N + 1) / (df[term] + 1)) + 1
        vector[term] = tf * idf

    return vector


vectors = [tfidf_vector(doc) for doc in documents]


def cosine(query_vector, doc_vector):

    common = set(query_vector) & set(doc_vector)

    numerator = sum(
        query_vector[t] * doc_vector[t]
        for t in common
    )

    q_norm = math.sqrt(
        sum(x * x for x in query_vector.values())
    )

    d_norm = math.sqrt(
        sum(x * x for x in doc_vector.values())
    )

    if q_norm == 0 or d_norm == 0:
        return 0

    return numerator / (q_norm * d_norm)


def tfidf_search(query):

    q_tokens = tokenize(query)
    q_vector = tfidf_vector(q_tokens)

    scores = [
        cosine(q_vector, vector)
        for vector in vectors
    ]

    return sorted(
        range(N),
        key=lambda i: scores[i],
        reverse=True
    )[:K]


# ---------------------------------------------------------
# RELEVANCE LABELS
# ---------------------------------------------------------

relevance = {}

for query, results in judgments.items():

    relevance[query] = {}

    for item in results:
        relevance[query][item["doc_id"]] = bool(
            item["relevant"]
        )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def precision_at_k(ranked_ids, labels):

    relevant = sum(
        1 for doc_id in ranked_ids
        if labels.get(doc_id, False)
    )

    return relevant / K


def recall_at_k(ranked_ids, labels):

    total_relevant = sum(
        1 for value in labels.values()
        if value
    )

    if total_relevant == 0:
        return 0

    retrieved_relevant = sum(
        1 for doc_id in ranked_ids
        if labels.get(doc_id, False)
    )

    return retrieved_relevant / total_relevant


print()
print("=" * 75)
print("EVIDENCERANK — BASELINE EVALUATION")
print("=" * 75)

baseline_p = []
baseline_r = []

for query, labels in relevance.items():

    ranked_indices = tfidf_search(query)

    ranked_ids = [
        papers[i]["id"]
        for i in ranked_indices
    ]

    p = precision_at_k(ranked_ids, labels)
    r = recall_at_k(ranked_ids, labels)

    baseline_p.append(p)
    baseline_r.append(r)

    print()
    print("Query:", query)
    print("P@5:", round(p, 3))
    print("Recall@5:", round(r, 3))

print()
print("-" * 75)

print(
    "Mean P@5:",
    round(sum(baseline_p) / len(baseline_p), 3)
)

print(
    "Mean Recall@5:",
    round(sum(baseline_r) / len(baseline_r), 3)
)

print("=" * 75)
print("TF-IDF BASELINE COMPLETE")
print("=" * 75)