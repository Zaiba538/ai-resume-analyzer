from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

def analyze_resume(resume_text):

    prompt = f"""
    Analyze this resume.

    Extract:
    - Skills
    - Education
    - Experience
    - Projects
    - Certifications

    Resume:
    {resume_text}
    """

    response = llm.invoke(prompt)

    return response.content