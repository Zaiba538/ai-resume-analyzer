"""TF-IDF keyword-match baseline: an exact port of legacy ``utils/matching.py``.

Kept for continuity with the legacy "X% Match" and as the evaluation baseline
(approved Rev. 3). ``calculate_match_score`` reproduces the legacy function
line for line: a default ``TfidfVectorizer()`` (lower-casing, token pattern
``(?u)\\b\\w\\w+\\b``, no stop-word removal, L2 norm, smoothed IDF) fitted on
exactly ``[resume_text, job_description]``, cosine similarity of the two rows,
multiplied by 100 and passed to ``round(score, 2)``. No other preprocessing.

Like the legacy code, it raises ``ValueError`` when neither text contains a
token of two or more word characters (an empty vocabulary). The API uses
``tfidf_keyword_match``, which maps exactly that case to 0.0, the cosine
similarity of two texts that share no terms.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMPTY_VOCABULARY = "empty vocabulary"


def calculate_match_score(resume_text, job_description):  # untyped, as in the legacy code
    docs = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(docs)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])

    score = similarity[0][0] * 100

    return round(score, 2)


def tfidf_keyword_match(resume_text: str, job_description: str) -> float:
    """Legacy score as a plain float; an empty shared vocabulary scores 0.0."""
    try:
        return float(calculate_match_score(resume_text, job_description))
    except ValueError as exc:
        if EMPTY_VOCABULARY in str(exc):
            return 0.0
        raise
