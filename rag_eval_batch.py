"""
Evaluate un-evaluated query_log rows for faithfulness and relevancy.
"""

# load requires packages...
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

CHUNK_DELIMITER = "\n--CHUNK--\n"


class BatchEval:
    """
    Runs faithfulness and relevancy evaluation concurrently via
    BatchEvalRunner, on rows fetched from query_log by db_logger.

    Attributes:
                    runner: LlamaIndex BatchEvalRunner, wired with both evaluators,
                    using the same LLM configured for answer generation as judge.
    """

    def __init__(self):
        """
        Load the HF token, configure the judge LLM, and build the batch
        runner with both evaluators.
        """

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
        """
        Fetch up to 15 un-evaluated query_log rows and evaluate them for
        faithfulness and relevancy. Only runs if at least 15 rows are
        available, so evaluation happens in full batches, not partial ones.

        Returns:
                        A list of dicts ready for db_logger.insert_eval_results(), or
                        an empty list if fewer than 15 rows are currently un-evaluated.
        """
        nest_asyncio.apply()

        rows = fetch_unevaluated()
        if not rows:
            print("[INTO] No unevaluated rows fetched")
            return []

        if len(rows) < 15:
            print(
                f"[INFO] Only {len(rows)} un-evaluated rows — waiting for a full batch of 15."
            )
            return []

        # get the data from all rows in list
        row_ids = [r[0] for r in rows]
        queries = [r[1] for r in rows]
        answer = [r[2] for r in rows]

        # split the chuck strings based delimiter
        contexts_list = [r[3].split(CHUNK_DELIMITER) if r[3] else [] for r in rows]

        # evaluate the responses
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
        return results
