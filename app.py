from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pdfplumber
import re

st.title("AI Resume Analyzer")
st.sidebar.title("About")

st.sidebar.info(
    "This AI Resume Analyzer compares resumes with job descriptions using NLP and Machine Learning."
)
job_description = st.text_area(
    "Paste Job Description"
)

uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

skills = [
    "python",
    "java",
    "sql",
    "mysql",
    "postgresql",
    "html",
    "css",
    "javascript",
    "react",
    "machine learning",
    "data structures",
    "algorithms"
]

if uploaded_file is not None:

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    cleaned_text = text.lower()

    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)

    detected_skills = []

    for skill in skills:
        if skill in cleaned_text:
            detected_skills.append(skill)

    st.success("Resume uploaded successfully!")

    st.subheader("Detected Skills")

    st.write(detected_skills)

    
    if job_description:
        documents = [cleaned_text, job_description.lower()]

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(documents)

        similarity_score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
        )

        score = similarity_score[0][0] * 100

        st.subheader("ATS Match Score")

        st.write(str(round(score, 2)) + "%")
        job_skills = []
        for skill in skills:
            if skill in job_description.lower():
                job_skills.append(skill)
        missing_skills = []
        for skill in job_skills:
            if skill not in detected_skills:
                missing_skills.append(skill)

        st.subheader("Missing Skills")
        st.write(missing_skills)
        st.write("Project Updated")


    