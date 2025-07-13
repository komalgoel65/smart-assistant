# backend_free.py

import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load PDF content
def load_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# Summarize text
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:1024], max_length=120, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Get answer using basic semantic similarity
def get_qa_answers(context_text, question):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = context_text.split(". ")
    question_embedding = model.encode(question, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
    best_score_idx = scores.argmax().item()

    return sentences[best_score_idx], sentences[best_score_idx]

# Generate basic logic questions
def generate_logic_questions(text):
    lines = text.strip().split("\n")
    selected = lines[:10] if len(lines) >= 10 else lines
    questions = []

    for line in selected:
        if len(line.split()) > 7:
            questions.append("What does the following mean? → " + line.strip())
        if len(questions) >= 3:
            break

    if not questions:
        questions = ["What is the main topic of the document?",
                     "What is the conclusion stated?",
                     "Explain any process described in the document."]
    return questions
