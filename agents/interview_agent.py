from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

def generate_questions(resume_analysis):

    prompt = f"""
    Generate interview questions based on:

    {resume_analysis}

    Include:
    - Beginner
    - Intermediate
    - Advanced
    """

    response = llm.invoke(prompt)

    return response.content