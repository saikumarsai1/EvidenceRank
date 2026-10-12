import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load papers
with open("data/raw/papers.json", "r", encoding="utf-8") as file:
    papers = json.load(file)


# Prepare documents
documents = []

for paper in papers:
    text = paper["title"] + " " + paper["abstract"]
    documents.append(text)


# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer()

# Search function
def search(query, top_k=3):

    # Convert query into TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    scores = cosine_similarity(query_vector, document_vectors)[0]

    # Get highest scoring documents
    ranked_indices = scores.argsort()[::-1][:top_k]

    # Display results
    for index in ranked_indices:
        print(
            papers[index]["id"],
            "-",
            papers[index]["title"],
            "| Score:",
            round(scores[index], 3)
        )


# Take user query
query = input("Enter your search query: ")

print("\nTop results:")
search(query)