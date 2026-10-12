import os
import sys


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from app import EvidenceRankApp
    from PySide6.QtWidgets import QApplication

    application = QApplication(sys.argv)
    application.setApplicationName("EvidenceRank")

    window = EvidenceRankApp()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
    