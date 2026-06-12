from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

def skill_gap_analysis(
    resume_text,
    job_description
):

    prompt = f"""
    Compare the resume with job description.

    Find:
    - Matching skills
    - Missing skills
    - Recommendations

    Resume:
    {resume_text}

    Job Description:
    {job_description}
    """

    response = llm.invoke(prompt)

    return response.content