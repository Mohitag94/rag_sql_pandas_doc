"""
Builds the index and logs the corresponding response or error to the database
along with the batch evaluation of the responses.
"""

# load requires packages...
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from db_logger import eval_logger, query_logger
from query_engine import IndexLoader, QueryEngine
from rag_eval_batch import BatchEval

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup and shutdown. Loads the persisted index,
    builds the query engine and batch evaluator, and stores both in
    app_state so every request reuses them instead of reloading per call.

    Raises:
            Exception: re-raised if index/engine loading fails, so the
            server does not start in a broken state.
    """
    try:
        # index load
        loader = IndexLoader()
        # build and store the query_engine
        app_state["query_engine"] = QueryEngine(loader.index)
        # built and store the evaluator
        app_state["batch_evaluator"] = BatchEval()
        print("\n[INFO] RAG Model for Pandas Loaded!!")

    except Exception as e:
        print("fail")
        raise e  # noqa: TRY201
    yield

    app_state.clear()


app = FastAPI(title="Pandas API Engine", lifespan=lifespan)


@app.get("/health")
def health_check():
    """
    Verify if the cached cloud RAG engine is actively mounted in memory.
    """
    if "query_engine" in app_state and app_state["query_engine"] is not None:
        return {
            "status": "healthy",
            "message": "RAG engine is fully loaded and ready to answer queries.",
            "cached_keys": list(app_state.keys()),
        }

    return {
        "status": "unhealthy",
        "message": "Application is running, but the RAG engine is missing from memory cache.",
    }


class QueryRequest(BaseModel):
    question: str


def evaluation():
    """
    Check whether enough un-evaluated query_log rows have accumulated,
    and if so, run batch evaluation and record the results.
    """
    evaluator = app_state.get("batch_evaluator")
    if not evaluator:
        print("[WARNING] Batch evaluator instance is not ready.")
        return

    # Call it on the instance variable
    results_eval = evaluator.eval()
    eval_logger(results_eval)


@app.post("/api/v1/ask")
async def handle_request(payload: QueryRequest, background_task: BackgroundTasks):
    """
    Answer a question using the pre-loaded RAG engine, then log the
    query and (conditionally) run batch evaluation as background tasks
    so the response returns to the caller without waiting on either.

    Args:
            payload: The incoming request body, containing the question.
            background_task: FastAPI's background task queue, used to run
            logging and evaluation after the response has been sent.

    Returns:
            A dict with the generated answer.

    Raises:
            HTTPException: 503 if the RAG engine isn't loaded in app_state.
    """
    # Grab the pre-loaded engine instantly from your memory dictionary
    query_engine = app_state.get("query_engine")
    if not query_engine:
        raise HTTPException(status_code=503, detail="RAG system engine is offline.")

    # Execute the query fast
    result = query_engine.ask(payload.question)

    background_task.add_task(query_logger, result)
    background_task.add_task(evaluation)
    return {"answer": result["answer"]}
