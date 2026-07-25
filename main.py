"""
Load the pandas documents and build a FAISS-backed index
and store in the local disk.
- Tokensier & Embedding Model: BAAI/bge-small-en-v1.5
- Chunk Size: 450 (under model's 512-token hard limit)
- Vector Storage Backend: FAIS
"""

# loading requires packages...
from pathlib import Path

import faiss
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    set_global_tokenizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# from dotenv import load_dotenv
from transformers import AutoTokenizer


class IndexBuilder:
    """
    Builds and persists a FAISS-backed vector index from the pandas
    documentation .rst files.

    Attributes:
                documents: Loaded Document objects, set by .load().
                index: The built VectorStoreIndex, set by .index_build().
    """

    def __init__(self, embed_dim=384):
        """
        Set up paths and configure the global tokenizer, embedding model,
        and chunk size before any documents are loaded or indexed.

        Args:
                        embed_dim: Output dimension of the embedding model. Must match
                        BAAI/bge-small-en-v1.5's actual output size (384) or FAISS
                        will raise a dimension-mismatch error.
        """
        # parent/root directory
        self.ROOT_DIR = Path(__file__).resolve().parent
        # data directory
        self.DATA_DIR = self.ROOT_DIR / "Data"
        # index storage/persist
        self.PERSIST_DIR = self.ROOT_DIR / "Storage"
        # embedding dimesion
        self.embed_dim = embed_dim
        self.documents = None
        self.index = None

        # configure tokensier & embedding model + chunk size & overlap
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5").encode
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.chunk_size = 450
        Settings.chunk_overlap = 50

    def load(self):
        """
        Load .rst documents from the disk.

        Returns:
            The list of loaded Document objects.
        """

        print("[INFO] Loading Documents...")
        self.documents = SimpleDirectoryReader(
            input_dir=self.DATA_DIR, required_exts=[".rst"]
        ).load_data()
        print("\tDone.")

        return self.documents

    def index_build(self):
        """
        Single VectorStoreIndex call for chuck, embedding & storage via FAISS

        Returns:
                The built VectorStoreIndex.

        Raises:
                ValueError: If .load() hasn't been called yet.
        """

        if self.documents is None:
            raise ValueError("No documents loaded — call .load() first.")

        print("[INFO] Indexing Documents...")
        faiss_index = faiss.IndexFlatL2(self.embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            self.documents, storage_context=storage_context, show_progress=True
        )
        print(
            f"\tDone.\n\t[INFO] Indexed {len(self.index.docstore.docs)} chunks into FAISS."
        )
        return self.index

    def persist(self):
        """
        Save the built index to disk so it doesn't need rebuilding every run

        Raises:
                ValueError: If .index_build() hasn't been called yet.
        """
        if self.index is None:
            raise ValueError("No index built — call .build_index() first.")

        print("[INFO] Storing the Index...")
        self.index.storage_context.persist(persist_dir=self.PERSIST_DIR)
        print(f"\tDone.\n\t[Info] Index saved to {self.PERSIST_DIR}")


if __name__ == "__main__":
    # load_dotenv()
    builder = IndexBuilder()
    builder.load()
    builder.index_build()
    builder.persist()
