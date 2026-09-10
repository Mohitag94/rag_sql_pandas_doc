"""
RAG model configuration file for model names and settings used.
"""

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 50


def config_embedding_tokenizer():
    """
    Set the global tokenizer and embedding models
    used to create/load RAG model.

                                                                    Args:
                                                                    hf_token: huggingface token for llm
    """
    import os

    from dotenv import load_dotenv
    from llama_index.core import Settings, set_global_tokenizer
    from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
    from transformers import AutoTokenizer

    load_dotenv()
    #  configure tokensier & embedding model
    set_global_tokenizer(AutoTokenizer.from_pretrained(EMBED_MODEL_NAME).encode)
    # Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        token=os.getenv("HF_TOKEN"),
    )


def config_chuck():
    """
    Set chucking and overlap sizes.
    Used during building rag model.
    """

    from llama_index.core import Settings

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def config_llm(hf_token):
    """
    Set the generation llm for respone to the query.

    Args:
                                                                    hf_token: huggingface token for llm
    """

    from llama_index.core import Settings
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

    Settings.llm = HuggingFaceInferenceAPI(
        model_name=LLM_MODEL_NAME,
        token=hf_token,
        temperature=0.2,
        max_tokens=256,
        provider="auto",
    )
