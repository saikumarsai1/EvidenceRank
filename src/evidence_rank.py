import json
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PAPERS_FILE = "data/raw/papers.json"


# Load papers
with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

if not papers:
    print("ERROR: No papers found.")
    exit()


# Prepare documents
documents = []

for paper in papers:
    title = str(paper.get("title", ""))
    abstract = str(paper.get("abstract", ""))

    documents.append(title + " " + abstract)


# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(documents)


def normalize(values):

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0.0] * len(values)

    return [
        (x - minimum) / (maximum - minimum)
        for x in values
    ]


# Authority = citation score
citation_values = []

for paper in papers:

    citations = paper.get("citations", 0)

    try:
        citations = float(citations)
    except:
        citations = 0

    citation_values.append(math.log1p(citations))


authority_scores = normalize(citation_values)


# Recency score
current_year = 2026

recency_raw = []

for paper in papers:

    try:
        year = int(paper.get("year", current_year))
    except:
        year = current_year

    age = max(0, current_year - year)

    recency_raw.append(1 / (1 + age))


recency_scores = normalize(recency_raw)


def search(query, top_k=5):

    # Query vector
    query_vector = vectorizer.transform([query])

    # Relevance
    relevance_raw = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    relevance_scores = normalize(
        relevance_raw.tolist()
    )


    # Final score
    final_scores = []

    for i in range(len(papers)):

        score = (
            0.60 * relevance_scores[i]
            + 0.20 * authority_scores[i]
            + 0.20 * recency_scores[i]
        )

        final_scores.append(score)


    # Rank
    ranked_indices = sorted(
        range(len(papers)),
        key=lambda i: final_scores[i],
        reverse=True
    )


    # TOP 5
    ranked_indices = ranked_indices[:5]


    print()
    print("=" * 80)
    print("EVIDENCERANK RESULTS")
    print("=" * 80)

    print("Query:", query)
    print()


    for rank, i in enumerate(ranked_indices, 1):

        paper = papers[i]

        print(f"#{rank} {paper.get('title', 'Untitled')}")
        print(f"ID: {paper.get('id', 'Unknown')}")
        print(f"Final Score : {final_scores[i]:.3f}")
        print(f"Relevance   : {relevance_scores[i]:.3f}")
        print(f"Authority   : {authority_scores[i]:.3f}")
        print(f"Recency     : {recency_scores[i]:.3f}")
        print(f"Year        : {paper.get('year', 'Unknown')}")
        print(f"Citations   : {paper.get('citations', 0)}")
        print()


if __name__ == "__main__":

    query = input("Enter your search query: ")

    search(query, top_k=5)