import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


# ============================================================
# 1. LOAD PAPERS
# ============================================================

with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


# ============================================================
# 2. DOCUMENT FIELDS
# ============================================================

titles = [paper["title"] for paper in papers]

abstracts = [paper["abstract"] for paper in papers]

full_text = [
    paper["title"] + " " + paper["abstract"]
    for paper in papers
]


# ============================================================
# 3. TF-IDF
# ============================================================

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_vectors = tfidf_vectorizer.fit_transform(
    full_text
)


# ============================================================
# 4. BM25
# ============================================================

tokenized_documents = [
    preprocess(text)
    for text in full_text
]

bm25 = BM25Okapi(tokenized_documents)


# ============================================================
# 5. TITLE / ABSTRACT ZONES
# ============================================================

title_vectorizer = TfidfVectorizer(
    stop_words="english"
)

abstract_vectorizer = TfidfVectorizer(
    stop_words="english"
)

title_vectors = title_vectorizer.fit_transform(titles)

abstract_vectors = abstract_vectorizer.fit_transform(abstracts)


# ============================================================
# 6. NORMALIZATION
# ============================================================

def normalize(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.ones(len(values))

    return (values - minimum) / (maximum - minimum)


# ============================================================
# 7. AUTHORITY + RECENCY
# ============================================================

citation_scores = normalize(
    [paper["citations"] for paper in papers]
)

CURRENT_YEAR = 2026

recency_values = np.array([
    1 / (CURRENT_YEAR - paper["year"] + 1)
    for paper in papers
])

recency_scores = normalize(recency_values)


# ============================================================
# 8. PHRASE MATCH
# ============================================================

def phrase_match(query, document):

    query = query.lower().strip()

    document = document.lower()

    if not query:
        return 0.0

    return 1.0 if query in document else 0.0


# ============================================================
# 9. EVIDENCERANK
# ============================================================

def search(
    query,
    top_k=5,
    min_year=None,
    min_citations=None
):

    # --------------------------------------------------------
    # Query vectors
    # --------------------------------------------------------

    tfidf_query = tfidf_vectorizer.transform([query])

    title_query = title_vectorizer.transform([query])

    abstract_query = abstract_vectorizer.transform([query])


    # --------------------------------------------------------
    # TF-IDF relevance
    # --------------------------------------------------------

    tfidf_scores = cosine_similarity(
        tfidf_query,
        tfidf_vectors
    )[0]


    # --------------------------------------------------------
    # BM25 relevance
    # --------------------------------------------------------

    query_tokens = preprocess(query)

    bm25_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype=float
    )

    bm25_scores = normalize(bm25_scores)


    # --------------------------------------------------------
    # Zone relevance
    # --------------------------------------------------------

    title_scores = cosine_similarity(
        title_query,
        title_vectors
    )[0]

    abstract_scores = cosine_similarity(
        abstract_query,
        abstract_vectors
    )[0]


    # --------------------------------------------------------
    # Scientific zone weighting
    # --------------------------------------------------------

    zone_scores = (
        0.70 * title_scores +
        0.30 * abstract_scores
    )


    # --------------------------------------------------------
    # Phrase bonus
    # --------------------------------------------------------

    phrase_scores = np.array([
        phrase_match(query, text)
        for text in full_text
    ])


    # --------------------------------------------------------
    # Final EvidenceRank
    # --------------------------------------------------------

    relevance_scores = (
        0.40 * normalize(tfidf_scores) +
        0.30 * bm25_scores +
        0.30 * normalize(zone_scores)
    )


    final_scores = (
        0.70 * relevance_scores +
        0.15 * citation_scores +
        0.10 * recency_scores +
        0.05 * phrase_scores
    )


    # --------------------------------------------------------
    # Metadata filtering
    # --------------------------------------------------------

    eligible = np.ones(len(papers), dtype=bool)

    if min_year is not None:

        eligible &= np.array([
            paper["year"] >= min_year
            for paper in papers
        ])

    if min_citations is not None:

        eligible &= np.array([
            paper["citations"] >= min_citations
            for paper in papers
        ])


    # Remove filtered documents
    filtered_scores = final_scores.copy()

    filtered_scores[~eligible] = -np.inf


    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        filtered_scores
    )[::-1][:top_k]


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("                    EVIDENCERANK")
    print("=" * 80)

    print(f"Query: {query}")

    if min_year is not None:
        print(f"Year >= {min_year}")

    if min_citations is not None:
        print(f"Citations >= {min_citations}")

    print("-" * 80)


    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        if filtered_scores[index] == -np.inf:
            continue

        print(
            f"\n#{rank}  {papers[index]['id']}"
        )

        print(
            f"Title: {papers[index]['title']}"
        )

        print(
            f"Score: {final_scores[index]:.3f}"
        )

        print(
            f"Relevance: {relevance_scores[index]:.3f}"
        )

        print(
            f"Authority: {citation_scores[index]:.3f}"
        )

        print(
            f"Recency: {recency_scores[index]:.3f}"
        )

        print(
            f"Year: {papers[index]['year']} | "
            f"Citations: {papers[index]['citations']}"
        )

        print(
            f"Title match: {title_scores[index]:.3f} | "
            f"Abstract match: {abstract_scores[index]:.3f}"
        )

        print(
            f"Phrase match: {phrase_scores[index]:.0f}"
        )

    print("\n" + "=" * 80)


# ============================================================
# 10. INTERACTIVE SEARCH
# ============================================================

query = input(
    "Enter your scientific search query: "
)

search(
    query,
    top_k=5
)