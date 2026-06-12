import streamlit as st
import tempfile

from utils.pdf_loader import extract_text_from_pdf
from utils.docx_loader import extract_text_from_docx
from utils.matching import calculate_match_score

from agents.resume_agent import analyze_resume
from agents.interview_agent import generate_questions
from agents.skill_gap_agent import skill_gap_analysis

st.set_page_config(
    page_title="AI Resume Assistant",
    layout="wide"
)

st.title("AI Resume Analyzer & Interview Generator")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if uploaded_file:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{file_extension}"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if file_extension == "pdf":

        resume_text = extract_text_from_pdf(
            temp_path
        )

    elif file_extension == "docx":

        resume_text = extract_text_from_docx(
            temp_path
        )

    else:
        st.error("Unsupported file type")
        st.stop()

    # Debugging output
    st.subheader("Extracted Resume Text")
    st.write(resume_text[:1000])

    # Resume Analysis
    st.subheader("Resume Analysis")

    analysis = analyze_resume(
        resume_text
    )

    st.write(analysis)

    # Interview Questions
    st.subheader("Interview Questions")

    questions = generate_questions(
        analysis
    )

    st.write(questions)

    # Match Score + Skill Gap
    if job_description.strip():

        score = calculate_match_score(
            resume_text,
            job_description
        )

        st.subheader("Resume Match Score")
        st.success(f"{score}% Match")

        st.subheader("Skill Gap Analysis")

        gap = skill_gap_analysis(
            resume_text,
            job_description
        )

        st.write(gap)