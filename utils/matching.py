from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(
    resume_text,
    job_description
):

    docs = [
        resume_text,
        job_description
    ]

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(docs)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )

    score = similarity[0][0] * 100

    return round(score, 2)