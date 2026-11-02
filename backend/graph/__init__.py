"""LangGraph workflows (approved Rev. 3, M7): the resume analysis graph.

Graphs are compiled without a checkpointer, so workflow state, which holds the
sanitised resume and job description during a run, is never persisted.
"""
