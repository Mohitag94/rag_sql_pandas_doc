"""
Evaluate un-evaluated query_log rows for faithfulness and relevancy.
"""

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

# nest_asyncio.apply()

CHUNK_DELIMITER = "\n--CHUNK--\n"


class BatchEval:
    def __init__(self):
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
        nest_asyncio.apply()

        rows = fetch_unevaluated()
        if not rows:
            print("[INTO] No unevaluated rows fetched")
            return []

        row_ids = [r[0] for r in rows]
        queries = [r[1] for r in rows]
        answer = [r[2] for r in rows]
        contexts_list = [r[3].split(CHUNK_DELIMITER) if r[3] else [] for r in rows]
        eval_resutl = self.runner.evaluate_response_strs(
            queries=queries, response_strs=answer, contexts_list=contexts_list
        )

        faith_results = eval_resutl["faithfulness"]
        relevancy_results = eval_resutl["relevancy"]

        results = [
            {
                "query_log_id": row_id,
                "faithfulness_passing": faith.passing,
                "faithfulness_feedback": faith.feedback,
                "relevancy_passing": relevancy.passing,
                "relevancy_feedback": relevancy.feedback,
            }
            for row_id, faith, relevancy in zip(
                row_ids, faith_results, relevancy_results
            )
        ]
        # print(results)
        return results
