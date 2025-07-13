# app_free.py

import streamlit as st
import os
from backend import load_pdf_text, summarize_text, get_qa_answers, generate_logic_questions

st.set_page_config(page_title="Free Smart Assistant", layout="wide")
st.title("📄 Free Smart Assistant for Document Summarization and Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = load_pdf_text(file_path)
    st.subheader("🔍 Auto Summary")
    summary = summarize_text(text)
    st.info(summary)

    mode = st.radio("Choose Interaction Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_question = st.text_input("Ask a question based on the document:")
        if user_question:
            answer, context = get_qa_answers(text, user_question)
            st.success(f"Answer: {answer}")
            st.markdown("**Justification (Context):**")
            st.markdown(f"> {context}")

    elif mode == "Challenge Me":
        st.write("Generating 3 logic/comprehension questions...")
        questions = generate_logic_questions(text)
        user_answers = []

        for idx, q in enumerate(questions):
            ans = st.text_input(f"Q{idx+1}: {q}")
            user_answers.append((q, ans))

        if st.button("Submit Answers"):
            st.markdown("### 📝 Feedback:")
            for q, ans in user_answers:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**Your Answer:** {ans}")
                st.markdown(f"✅ Model's Expected Answer: [Use context manually or add local logic if needed]")
