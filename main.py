"""
Day 4-5 starter prototype.
Goal: prove the RAG pipeline works end to end on LlamaIndex's official
sample data (Paul Graham essay) before touching the pandas docs.

Run with: uv run python main.py
"""

import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Embeddings run locally, free, no API call
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Generation calls the Hugging Face Inference API
Settings.llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    token=hf_token,
    temperature=0.2,
    max_tokens=256,
    provider="auto",
)

# Load and index the sample data
documents = SimpleDirectoryReader(
    "/Users/mohitag/Documents/Projects/RAG_SQL_Project/Data"
).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Query it
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What did the author do growing up?")

print("\n--- ANSWER ---")
print(response)
