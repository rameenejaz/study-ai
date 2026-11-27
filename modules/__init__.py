"""StudyMate AI backend modules.

Each module is importable without Streamlit so that it can be unit-tested
on its own. The Transformer models are only imported when they are actually
loaded (see ``modules.models``), so the test-suite and the classical ML parts
work even on machines without PyTorch.
"""

__all__ = [
    "classification",
    "extraction",
    "facts",
    "flashcards",
    "mcqs",
    "models",
    "preprocessing",
    "summarization",
]
