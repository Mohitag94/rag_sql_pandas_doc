import os

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.evaluation import (
    BatchEvalRunner,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

from config import config_llm
from db_logger import fetch_unevaluated

nest_asyncio.apply()

CHUNK_DELIMITER = "\n--CHUNK--\n"


class BatchEval:
    def __init__(
        self,
    ):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        config_llm(hf_token)

        self.runner = BatchEvalRunner(
            {
                "faithfulness": FaithfulnessEvaluator(llm=Settings.llm),
                "relevancy": RelevancyEvaluator(llm=Settings.llm),
            },
            workers=4,
        )

    def eval(self):
        rows = fetch_unevaluated()
        return {}
