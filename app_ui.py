import requests
import streamlit as st

API_URL = "https://pandas-rag-api-mka-576093593374.us-central1.run.app/api/v1/ask"


st.title("Pandas Docs Q&A")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about pandas:")

if st.button("Ask"):
    response = requests.post(API_URL, json={"question": question})
    result = response.json()
    st.session_state.history.insert(
        0,
        {
            "question": question,
            "answer": result.get("answer"),
            "confidence_score": result.get("confidence_score"),
        },
    )
    st.session_state.history = st.session_state.history[:5]

for entry in st.session_state.history:
    st.write(f"**Q:** {entry['question']}")
    st.write(f"**A:** {entry['answer']}")
    if entry["confidence_score"] is not None:
        st.metric("Confidence Score", f"{entry['confidence_score']:.3f}")
    st.divider()
