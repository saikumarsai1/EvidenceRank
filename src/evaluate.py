import json
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILES
# ============================================================

PAPERS_FILE = "data/raw/papers.json"
JUDGMENTS_FILE = "data/raw/judgments.json"


# ============================================================
# QUERIES
# ============================================================

QUERIES = [
    "air pollution cardiovascular disease",
    "air pollution respiratory disease",
    "climate change human health",
    "particulate pollution health",
    "air pollution mortality"
]


# ============================================================
# LOAD PAPERS
# ============================================================

with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)


if not papers:
    print("ERROR: papers.json is empty.")
    exit()


# ============================================================
# PREPARE DOCUMENTS
# ============================================================

documents = []

for paper in papers:

    title = str(paper.get("title", ""))
    abstract = str(paper.get("abstract", ""))

    documents.append(title + " " + abstract)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(documents)


# ============================================================
# NORMALIZE
# ============================================================

def normalize(values):

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0.0] * len(values)

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


# ============================================================
# AUTHORITY SCORE
# ============================================================

citation_values = []

for paper in papers:

    try:
        citations = float(
            paper.get("citations", 0)
        )
    except:
        citations = 0.0

    citation_values.append(
        math.log1p(max(0, citations))
    )


authority_scores = normalize(
    citation_values
)


# ============================================================
# RECENCY SCORE
# ============================================================

CURRENT_YEAR = 2026

recency_values = []

for paper in papers:

    try:
        year = int(
            paper.get("year", CURRENT_YEAR)
        )
    except:
        year = CURRENT_YEAR

    age = max(
        0,
        CURRENT_YEAR - year
    )

    recency_values.append(
        1 / (1 + age)
    )


recency_scores = normalize(
    recency_values
)


# ============================================================
# EVIDENCERANK RETRIEVAL
# ============================================================

def retrieve(query, top_k=5):

    query_vector = vectorizer.transform(
        [query]
    )

    relevance_values = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    relevance_scores = normalize(
        relevance_values.tolist()
    )

    final_scores = []

    for i in range(len(papers)):

        score = (
            0.60 * relevance_scores[i]
            + 0.20 * authority_scores[i]
            + 0.20 * recency_scores[i]
        )

        final_scores.append(score)


    # Rank papers by final score
    ranked_indices = sorted(
        range(len(papers)),
        key=lambda i: final_scores[i],
        reverse=True
    )


    # Top 5
    ranked_indices = ranked_indices[:top_k]


    # Return document IDs
    return [
        papers[i]["id"]
        for i in ranked_indices
    ]


# ============================================================
# LOAD JUDGMENTS
# ============================================================

with open(
    JUDGMENTS_FILE,
    "r",
    encoding="utf-8"
) as file:

    judgments = json.load(file)


# ============================================================
# GET RELEVANT DOCUMENT IDs
# ============================================================

def get_relevant_ids(query):

    relevant_ids = set()


    # judgments.json structure:
    #
    # {
    #     "query": [
    #         {
    #             "rank": 1,
    #             "doc_id": "...",
    #             "relevant": true
    #         }
    #     ]
    # }

    query_judgments = judgments.get(
        query,
        []
    )


    for judgment in query_judgments:

        if not isinstance(
            judgment,
            dict
        ):
            continue


        if judgment.get(
            "relevant",
            False
        ):

            doc_id = judgment.get(
                "doc_id"
            )

            if doc_id:
                relevant_ids.add(
                    doc_id
                )


    return relevant_ids


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 80)
print("EVIDENCERANK EVALUATION")
print("=" * 80)


all_precision = []
all_recall = []


for query in QUERIES:

    # Retrieve top 5
    retrieved = retrieve(
        query,
        top_k=5
    )


    # Get human-judged relevant papers
    relevant = get_relevant_ids(
        query
    )


    # Count relevant retrieved papers
    relevant_retrieved = 0

    for doc_id in retrieved:

        if doc_id in relevant:
            relevant_retrieved += 1


    # ========================================================
    # PRECISION@5
    # ========================================================

    precision = (
        relevant_retrieved / 5
    )


    # ========================================================
    # RECALL@5
    # ========================================================

    if len(relevant) == 0:

        recall = 0.0

    else:

        recall = (
            relevant_retrieved
            / len(relevant)
        )


    all_precision.append(
        precision
    )

    all_recall.append(
        recall
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("Query:", query)

    print(
        "Retrieved:",
        retrieved
    )

    print(
        "Relevant retrieved:",
        relevant_retrieved
    )

    print(
        "Relevant total:",
        len(relevant)
    )

    print(
        f"P@5: {precision:.3f}"
    )

    print(
        f"Recall@5: {recall:.3f}"
    )


# ============================================================
# MEAN SCORES
# ============================================================

mean_precision = (
    sum(all_precision)
    / len(all_precision)
)

mean_recall = (
    sum(all_recall)
    / len(all_recall)
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    f"Mean P@5      : {mean_precision:.3f}"
)

print(
    f"Mean Recall@5 : {mean_recall:.3f}"
)

print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)