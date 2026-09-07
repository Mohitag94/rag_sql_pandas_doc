"""
Load the presisted FAISS index and answer question aganist it,
log every query to the Neon query_log table.
- LLM: meta-llama/Llama-3.1-8B-Instruct via HF Inference API
- Logging: every call to .query() writes a row to query_log
        (question, retrieved chunk ids, confidence, latency, error)
"""

# loading requires packages...


# from dotenv import load_dotenv
