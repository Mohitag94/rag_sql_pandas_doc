"""
Load the presisted FAISS index and answer question aganist it,
log every query to the Neon query_log table.
- LLM: meta-llama/Llama-3.1-8B-Instruct via HF Inference API
- Logging: every call to .query() writes a row to query_log
        (question, retrieved chunk ids, confidence, latency, error)
"""

# loading requires packages...
from pathlib import Path

import faiss
from llama_index.core import (
    Settings,
    StorageContext,
    local_index_from_storage,
    set_global_tokenizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# from dotenv import load_dotenv
from transformers import AutoTokenizer

class IndexLoader:

    def __init__(self, embed_dim=384):
        # parent/root directory
        self.ROOT_DIR = Path(__file__).resolve().parent
        # index storage/persist
        self.PERSIST_DIR = self.ROOT_DIR / "Storage"
        # embedding dimesion
        self.embed_dim = embed_dim
        self.index = None

        # configure tokensier & embedding model + chunk size & overlap
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5").encode
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.chunk_size = 450
        Settings.chunk_overlap = 50
        
        self._load_index()

    def _load_index(self):
        