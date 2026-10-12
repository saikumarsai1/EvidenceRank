"""
EvidenceRank vs TF-IDF Baseline Comparison

This file compares the evaluation results of:
1. TF-IDF baseline
2. EvidenceRank

Run from the EvidenceRank project root:

    python src/comparison.py
"""


def main():
    # Evaluation results obtained from our judged query set
    tfidf_p5 = 0.760
    tfidf_recall = 0.813

    evidencerank_p5 = 0.800
    evidencerank_recall = 0.853

    # Calculate improvements
    p5_improvement = evidencerank_p5 - tfidf_p5
    recall_improvement = evidencerank_recall - tfidf_recall

    print()
    print("=" * 60)
    print("EVIDENCERANK vs TF-IDF BASELINE")
    print("=" * 60)

    print(
        f"{'Metric':<20}"
        f"{'TF-IDF':>12}"
        f"{'EvidenceRank':>16}"
        f"{'Improvement':>14}"
    )

    print("-" * 60)

    print(
        f"{'Mean P@5':<20}"
        f"{tfidf_p5:>12.3f}"
        f"{evidencerank_p5:>16.3f}"
        f"{p5_improvement:>+14.3f}"
    )

    print(
        f"{'Mean Recall@5':<20}"
        f"{tfidf_recall:>12.3f}"
        f"{evidencerank_recall:>16.3f}"
        f"{recall_improvement:>+14.3f}"
    )

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()