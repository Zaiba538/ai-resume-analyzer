"""Server-side storage (approved Rev. 3, M7): the in-memory TTL report store only.

Only derived structured data is stored, in process memory, until it expires.
Raw resume text, job descriptions, previews and answers are never stored.
"""
