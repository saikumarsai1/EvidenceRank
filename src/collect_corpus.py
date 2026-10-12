import json
import urllib.parse
import urllib.request
import time


OUTPUT_FILE = "data/raw/papers.json"

QUERIES = [
    "air pollution health",
    "climate change human health",
    "respiratory disease environment",
    "cardiovascular disease air pollution"
]

PAPERS_PER_QUERY = 30

BASE_URL = "https://api.openalex.org/works"


def abstract_from_inverted_index(inverted_index):

    if not inverted_index:
        return ""

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(
        word for _, word in words
    )


def fetch_papers(query):

    print()
    print("Downloading:", query)

    params = {
        "search": query,
        "per-page": PAPERS_PER_QUERY,
        "sort": "cited_by_count:desc"
    }

    url = (
        BASE_URL
        + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EvidenceRank/1.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=60
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    results = []

    for work in data.get("results", []):

        title = work.get("title")

        if not title:
            continue

        abstract = abstract_from_inverted_index(
            work.get("abstract_inverted_index")
        )

        publication_year = (
            work.get("publication_year")
        )

        citations = (
            work.get("cited_by_count", 0)
        )

        openalex_id = work.get("id")

        landing_page = None

        primary_location = work.get(
            "primary_location"
        )

        if primary_location:

            landing_page = (
                primary_location.get(
                    "landing_page_url"
                )
            )

        results.append({

            "id": "OA-" + openalex_id.split("/")[-1],

            "title": title,

            "abstract": abstract,

            "year": publication_year,

            "citations": citations,

            "source": "OpenAlex",

            "url": landing_page or openalex_id
        })

    print(
        "Retrieved:",
        len(results)
    )

    return results


def main():

    all_papers = {}

    for query in QUERIES:

        try:

            papers = fetch_papers(query)

            for paper in papers:

                all_papers[
                    paper["id"]
                ] = paper

            time.sleep(1)

        except Exception as error:

            print()
            print("ERROR:")
            print(error)

    papers = list(
        all_papers.values()
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            papers,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("CORPUS COLLECTION COMPLETE")
    print("=" * 70)
    print(
        "Unique papers:",
        len(papers)
    )
    print(
        "Saved to:",
        OUTPUT_FILE
    )
    print("=" * 70)


if __name__ == "__main__":
    main()