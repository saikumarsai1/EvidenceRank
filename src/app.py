import sys
import json
import math

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QScrollArea,
    QStackedWidget,
    QProgressBar,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DATA
# ============================================================

PAPERS_FILE = "data/raw/papers.json"

with open(PAPERS_FILE, "r", encoding="utf-8") as f:
    PAPERS = json.load(f)


# ============================================================
# SEARCH ENGINE
# ============================================================

documents = []

for paper in PAPERS:
    title = str(paper.get("title", ""))
    abstract = str(paper.get("abstract", ""))
    documents.append(title + " " + abstract)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

document_vectors = vectorizer.fit_transform(documents)


def normalize(values):

    if not values:
        return []

    low = min(values)
    high = max(values)

    if high == low:
        return [0.0] * len(values)

    return [
        (x - low) / (high - low)
        for x in values
    ]


# Authority
citation_values = []

for paper in PAPERS:
    try:
        citations = float(paper.get("citations", 0))
    except:
        citations = 0

    citation_values.append(
        math.log1p(max(0, citations))
    )

authority_scores = normalize(citation_values)


# Recency
recency_values = []

for paper in PAPERS:

    try:
        year = int(paper.get("year", 2026))
    except:
        year = 2026

    age = max(0, 2026 - year)

    recency_values.append(
        1 / (1 + age)
    )

recency_scores = normalize(recency_values)


def search_papers(query, top_k=5):

    query_vector = vectorizer.transform([query])

    relevance_raw = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    relevance_scores = normalize(
        relevance_raw.tolist()
    )

    results = []

    for i, paper in enumerate(PAPERS):

        relevance = relevance_scores[i]
        authority = authority_scores[i]
        recency = recency_scores[i]

        final_score = (
            0.60 * relevance
            + 0.20 * authority
            + 0.20 * recency
        )

        results.append({
            "index": i,
            "paper": paper,
            "relevance": relevance,
            "authority": authority,
            "recency": recency,
            "score": final_score,
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# COLORS / STYLE
# ============================================================

BACKGROUND = "#0B1020"
CARD = "#141B2D"
CARD_2 = "#192238"
TEXT = "#F4F7FF"
MUTED = "#9AA7C0"
ACCENT = "#6C8CFF"
ACCENT_2 = "#8B7CFF"
BORDER = "#27324A"


STYLESHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}

QWidget {{
    color: {TEXT};
    font-family: "Segoe UI";
}}

QLabel {{
    background: transparent;
}}

QLineEdit {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px 18px;
    color: {TEXT};
    font-size: 15px;
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

QPushButton {{
    background: {ACCENT};
    border: none;
    border-radius: 12px;
    padding: 13px 22px;
    color: white;
    font-weight: 600;
    font-size: 14px;
}}

QPushButton:hover {{
    background: {ACCENT_2};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QProgressBar {{
    background: #222B40;
    border: none;
    border-radius: 5px;
    height: 8px;
}}

QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
"""


# ============================================================
# PAPER CARD
# ============================================================

class PaperCard(QFrame):

    def __init__(self, result, rank, open_callback):

        super().__init__()

        self.setStyleSheet(f"""
        QFrame {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 18px;
        }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        paper = result["paper"]

        # Rank
        rank_label = QLabel(f"#{rank}")
        rank_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 13px; font-weight: bold;"
        )

        layout.addWidget(rank_label)

        # Title
        title = QLabel(
            paper.get("title", "Untitled")
        )

        title.setWordWrap(True)
        title.setStyleSheet(
            "font-size: 18px; font-weight: 700;"
        )

        layout.addWidget(title)

        # Metadata
        year = paper.get("year", "N/A")
        citations = paper.get("citations", 0)

        metadata = QLabel(
            f"Research Paper   •   {year}   •   "
            f"{citations:,} citations"
        )

        metadata.setStyleSheet(
            f"color: {MUTED}; font-size: 13px;"
        )

        layout.addWidget(metadata)

        # Score
        score_layout = QHBoxLayout()

        score_text = QLabel(
            f"EvidenceRank  {result['score']:.3f}"
        )

        score_text.setStyleSheet(
            f"color: {TEXT}; font-weight: 700;"
        )

        score_layout.addWidget(score_text)

        score_layout.addStretch()

        view_button = QPushButton("View details")
        view_button.clicked.connect(
            lambda: open_callback(result)
        )

        score_layout.addWidget(view_button)

        layout.addLayout(score_layout)

        # Progress
        progress = QProgressBar()

        progress.setRange(0, 1000)
        progress.setValue(
            int(result["score"] * 1000)
        )

        progress.setTextVisible(False)

        layout.addWidget(progress)

        # Components
        components = QHBoxLayout()

        components.addWidget(
            self.metric(
                "Relevance",
                result["relevance"]
            )
        )

        components.addWidget(
            self.metric(
                "Authority",
                result["authority"]
            )
        )

        components.addWidget(
            self.metric(
                "Recency",
                result["recency"]
            )
        )

        layout.addLayout(components)

    def metric(self, name, value):

        label = QLabel(
            f"{name}\n{value:.3f}"
        )

        label.setAlignment(Qt.AlignCenter)

        label.setStyleSheet(
            f"""
            color: {MUTED};
            background: {CARD_2};
            border-radius: 10px;
            padding: 8px;
            font-size: 12px;
            """
        )

        return label


# ============================================================
# MAIN WINDOW
# ============================================================

class EvidenceRankApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "EvidenceRank — Research Intelligence"
        )

        self.resize(1250, 820)

        self.setMinimumSize(
            QSize(1050, 700)
        )

        self.setStyleSheet(STYLESHEET)

        self.stack = QStackedWidget()

        self.setCentralWidget(
            self.stack
        )

        self.search_page = self.create_search_page()

        self.details_page = self.create_details_page()

        self.stack.addWidget(
            self.search_page
        )

        self.stack.addWidget(
            self.details_page
        )


    # ========================================================
    # SEARCH PAGE
    # ========================================================

    def create_search_page(self):

        page = QWidget()

        main = QVBoxLayout(page)

        main.setContentsMargins(
            55, 40, 55, 35
        )

        main.setSpacing(20)


        # Header
        header = QHBoxLayout()

        brand = QLabel("EvidenceRank")

        brand.setStyleSheet(
            f"""
            font-size: 25px;
            font-weight: 800;
            color: {TEXT};
            """
        )

        header.addWidget(brand)

        header.addStretch()

        tag = QLabel(
            "SCIENCE  •  VERTICAL SEARCH"
        )

        tag.setStyleSheet(
            f"""
            color: {ACCENT};
            font-size: 11px;
            font-weight: 700;
            """
        )

        header.addWidget(tag)

        main.addLayout(header)


        # Hero
        hero = QLabel(
            "Research intelligence,\n"
            "ranked by evidence."
        )

        hero.setStyleSheet(
            """
            font-size: 38px;
            font-weight: 800;
            line-height: 1.2;
            """
        )

        main.addWidget(hero)


        subtitle = QLabel(
            "Discover scientific literature using relevance, "
            "citation authority and recency."
        )

        subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 15px;"
        )

        main.addWidget(subtitle)


        # Search
        search_row = QHBoxLayout()

        self.search_box = QLineEdit()

        self.search_box.setPlaceholderText(
            "Search research papers..."
        )

        self.search_box.returnPressed.connect(
            self.perform_search
        )

        search_row.addWidget(
            self.search_box
        )

        search_button = QPushButton(
            "Search  →"
        )

        search_button.setMinimumWidth(130)

        search_button.clicked.connect(
            self.perform_search
        )

        search_row.addWidget(
            search_button
        )

        main.addLayout(search_row)


        # Suggestions
        suggestions = QHBoxLayout()

        suggestion_text = QLabel(
            "Try:"
        )

        suggestion_text.setStyleSheet(
            f"color: {MUTED};"
        )

        suggestions.addWidget(
            suggestion_text
        )

        for text in [
            "Air pollution",
            "Climate change",
            "Cardiovascular disease",
        ]:

            button = QPushButton(text)

            button.setStyleSheet(
                f"""
                QPushButton {{
                    background: {CARD_2};
                    border: 1px solid {BORDER};
                    border-radius: 10px;
                    padding: 8px 12px;
                    color: {MUTED};
                }}
                QPushButton:hover {{
                    color: white;
                    border: 1px solid {ACCENT};
                }}
                """
            )

            button.clicked.connect(
                lambda checked=False, t=text:
                self.set_query(t)
            )

            suggestions.addWidget(
                button
            )

        suggestions.addStretch()

        main.addLayout(suggestions)


        # Results header
        result_header = QHBoxLayout()

        self.results_title = QLabel(
            "Top research papers"
        )

        self.results_title.setStyleSheet(
            "font-size: 20px; font-weight: 700;"
        )

        result_header.addWidget(
            self.results_title
        )

        result_header.addStretch()

        self.result_count = QLabel(
            f"{len(PAPERS)} papers indexed"
        )

        self.result_count.setStyleSheet(
            f"color: {MUTED};"
        )

        result_header.addWidget(
            self.result_count
        )

        main.addLayout(result_header)


        # Scroll
        self.scroll = QScrollArea()

        self.scroll.setWidgetResizable(
            True
        )

        self.results_container = QWidget()

        self.results_layout = QVBoxLayout(
            self.results_container
        )

        self.results_layout.setSpacing(14)

        self.results_layout.addStretch()

        self.scroll.setWidget(
            self.results_container
        )

        main.addWidget(
            self.scroll
        )

        return page


    # ========================================================
    # DETAILS PAGE
    # ========================================================

    def create_details_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        layout.setContentsMargins(
            55, 45, 55, 40
        )

        back = QPushButton(
            "← Back to results"
        )

        back.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {MUTED};
                padding: 8px;
            }}
            """
        )

        back.clicked.connect(
            lambda: self.stack.setCurrentWidget(
                self.search_page
            )
        )

        layout.addWidget(back)

        self.details_title = QLabel()

        self.details_title.setWordWrap(True)

        self.details_title.setStyleSheet(
            "font-size: 30px; font-weight: 800;"
        )

        layout.addWidget(
            self.details_title
        )

        self.details_metadata = QLabel()

        self.details_metadata.setStyleSheet(
            f"color: {MUTED}; font-size: 14px;"
        )

        layout.addWidget(
            self.details_metadata
        )

        # Score card
        score_card = QFrame()

        score_card.setStyleSheet(
            f"""
            QFrame {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 18px;
            }}
            """
        )

        score_layout = QVBoxLayout(
            score_card
        )

        score_layout.setContentsMargins(
            25, 22, 25, 22
        )

        score_heading = QLabel(
            "WHY THIS PAPER RANKED HERE"
        )

        score_heading.setStyleSheet(
            f"""
            color: {ACCENT};
            font-size: 11px;
            font-weight: 800;
            """
        )

        score_layout.addWidget(
            score_heading
        )

        self.detail_score = QLabel()

        self.detail_score.setStyleSheet(
            "font-size: 34px; font-weight: 800;"
        )

        score_layout.addWidget(
            self.detail_score
        )

        self.detail_components = QLabel()

        self.detail_components.setStyleSheet(
            f"color: {MUTED}; font-size: 14px;"
        )

        score_layout.addWidget(
            self.detail_components
        )

        layout.addWidget(
            score_card
        )

        # Abstract
        abstract_heading = QLabel(
            "Abstract"
        )

        abstract_heading.setStyleSheet(
            "font-size: 20px; font-weight: 700;"
        )

        layout.addWidget(
            abstract_heading
        )

        self.abstract_text = QLabel()

        self.abstract_text.setWordWrap(True)

        self.abstract_text.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 14px;
            line-height: 1.5;
            """
        )

        layout.addWidget(
            self.abstract_text
        )

        layout.addStretch()

        return page


    # ========================================================
    # SEARCH
    # ========================================================

    def set_query(self, text):

        self.search_box.setText(text)

        self.perform_search()


    def perform_search(self):

        query = self.search_box.text().strip()

        if not query:
            QMessageBox.information(
                self,
                "Search",
                "Please enter a research query."
            )
            return

        results = search_papers(
            query,
            top_k=5
        )

        self.results_title.setText(
            f"Results for “{query}”"
        )

        # Clear old results
        while self.results_layout.count():

            item = self.results_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()


        for rank, result in enumerate(
            results,
            start=1
        ):

            card = PaperCard(
                result,
                rank,
                self.show_details
            )

            self.results_layout.addWidget(
                card
            )

        self.results_layout.addStretch()

        self.result_count.setText(
            f"{len(results)} results"
        )


    # ========================================================
    # DETAILS
    # ========================================================

    def show_details(self, result):

        paper = result["paper"]

        self.details_title.setText(
            paper.get(
                "title",
                "Untitled"
            )
        )

        self.details_metadata.setText(
            f"{paper.get('year', 'N/A')}  •  "
            f"{paper.get('citations', 0):,} citations  •  "
            f"{paper.get('id', 'N/A')}"
        )

        self.detail_score.setText(
            f"EvidenceRank  {result['score']:.3f}"
        )

        self.detail_components.setText(
            f"Relevance   {result['relevance']:.3f}\n"
            f"Authority   {result['authority']:.3f}\n"
            f"Recency     {result['recency']:.3f}"
        )

        abstract = paper.get(
            "abstract",
            "No abstract available."
        )

        self.abstract_text.setText(
            abstract
        )

        self.stack.setCurrentWidget(
            self.details_page
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    app.setApplicationName(
        "EvidenceRank"
    )

    window = EvidenceRankApp()

    window.show()

    sys.exit(
        app.exec()
    )