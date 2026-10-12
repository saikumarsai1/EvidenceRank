import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


titles = [paper["title"] for paper in papers]
abstracts = [paper["abstract"] for paper in papers]


title_vectorizer = TfidfVectorizer(stop_words="english")
abstract_vectorizer = TfidfVectorizer(stop_words="english")


title_vectors = title_vectorizer.fit_transform(titles)
abstract_vectors = abstract_vectorizer.fit_transform(abstracts)


def search(query, top_k=3):

    title_query = title_vectorizer.transform([query])
    abstract_query = abstract_vectorizer.transform([query])

    title_scores = cosine_similarity(
        title_query, title_vectors
    )[0]

    abstract_scores = cosine_similarity(
        abstract_query, abstract_vectors
    )[0]

    final_scores = (
        0.70 * title_scores +
        0.30 * abstract_scores
    )

    ranked_indices = final_scores.argsort()[::-1][:top_k]

    print("\nEvidenceRank Zone Results:")

    for index in ranked_indices:
        print(
            papers[index]["id"],
            "-",
            papers[index]["title"],
            "| Score:",
            round(final_scores[index], 3),
            "| Title:",
            round(title_scores[index], 3),
            "| Abstract:",
            round(abstract_scores[index], 3)
        )


query = input("Enter your search query: ")
search(query)