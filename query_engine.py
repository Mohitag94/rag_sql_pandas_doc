"""
Load the presisted FAISS index and answer question aganist it,
log every query to the Neon query_log table.
- LLM: meta-llama/Llama-3.1-8B-Instruct via HF Inference API
- Logging: every call to .query() writes a row to query_log
        (question, retrieved chunk ids, confidence, latency, error)
"""

# loading requires packages...
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

from config import EMBED_DIM, config_embedding_tokenizer, config_llm


class IndexLoader:
    """
    Loads a custome build FAISS-based vector index from disk.

    Attributes:
        index: the loaded VectorStoreIndex
    """

    def __init__(self, embed_dim=EMBED_DIM):
        """
        Set up paths and configure the global tokenizer, embedding model,
        then loads the presist index.

        Agrs:
            embed_dim: Output dimension of the embedding model. Unused
            by loading itself, kept for consistency with IndexBuilder.
        """
        # parent/root directory
        self.ROOT_DIR = Path(__file__).resolve().parent
        # index storage/persist
        self.PERSIST_DIR = self.ROOT_DIR / "Storage"
        # embedding dimesion
        self.embed_dim = embed_dim
        self.index = None

        # configure tokensier & embedding model + chunk size & overlap
        config_embedding_tokenizer()

        self._load_index()

    def _load_index(self):
        """
        Loads the presisted FAISS-backed index from local disk.

        Return:
            index: the loaded VectorStoreIndex.
        """
        # loading the index from the local storage
        vector_store = FaissVectorStore.from_persist_dir(self.PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            persist_dir=self.PERSIST_DIR, vector_store=vector_store
        )
        self.index = load_index_from_storage(storage_context)


class QueryEngine:
    """
    Builts a query engine from the already loaded index and answer question
    against it, returns the answer alone with metadata.

    Attributes:
        query_engine: the LlamaIndex query engine.
    """

    def __init__(self, index):
        """
        Configures the llm responsible for answer generation and
        buils the query engine from the loaded index.

        Agrs:
            index: the loaded VectorStoreIndex.
        """
        # llm configuration
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        config_llm(self.hf_token)
        # building the query engine
        self.query_engine = index.as_query_engine()

    def ask(self, question: str) -> dict:
        """
        Ask a question against the loaded index, timing the call and
        extracting retrieval metadata alongside the answer.

        Agrs:
            question: the query to be answered.

        Returns:
            a dict with keys: answer, confidence, chunk_ids, latency_ms.
        """
        # generating an answer with build query engine
        start = time.perf_counter()
        response = self.query_engine.query(question)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # getting the top confidence score
        confidence = None
        if response.source_nodes:
            confidence = response.source_nodes[0].score

        # getting the chuck ids of the respons
        chunk_ids = [node.node_id for node in response.source_nodes]

        return {
            "answer": response.response,
            "confidence": confidence,
            "chuck_ids": chunk_ids,
            "latency_ms": latency_ms,
        }
