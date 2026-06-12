# AI Resume Analyzer & Interview Generator

## Overview

AI Resume Analyzer & Interview Generator is a multi-agent AI application built using Streamlit, LangChain, and Groq LLM. The system analyzes resumes, generates interview questions, identifies skill gaps, and calculates resume-job description match scores.

The application supports both PDF and DOCX resume formats, making it useful for job seekers who want to evaluate their resumes and prepare for interviews.

---

## Features

* Upload resumes in PDF format
* Upload resumes in DOCX format
* AI-powered resume analysis
* Interview question generation
* Resume-job description matching
* Skill gap analysis
* Multi-agent architecture
* Streamlit-based user interface
* Groq LLM integration

---

## Project Structure

```text
MULTI AGENTS/
│
├── agents/
│   ├── interview_agent.py
│   ├── resume_agent.py
│   └── skill_gap_agent.py
│
├── utils/
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   └── matching.py
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Tech Stack

* Python
* Streamlit
* LangChain
* Groq API
* Scikit-Learn
* PyPDF
* Python-Docx

---

## How It Works

1. User uploads a resume (PDF or DOCX).
2. Resume text is extracted.
3. Resume Agent analyzes skills, education, experience, projects, and certifications.
4. Interview Agent generates interview questions based on the resume.
5. Match Score module compares the resume with the provided job description.
6. Skill Gap Agent identifies missing skills and improvement areas.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd multi-agents
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

* LangGraph workflow orchestration
* Agent memory
* Tool calling support
* Authentication system
* Resume recommendations
* Database integration
* Deployment on Render and Hugging Face Spaces

---

## Author

Zaiba Ansari

Aspiring Data Scientist | Machine Learning Enthusiast | AI Developer
