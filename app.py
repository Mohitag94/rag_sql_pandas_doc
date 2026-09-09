from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from db_logger import eval_logger, query_logger
from query_engine import IndexLoader, QueryEngine
from rag_eval_batch import BatchEval

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loader = IndexLoader()
        app_state["query_engine"] = QueryEngine(loader.index)
        app_state["batch_evaluator"] = BatchEval()
        print("\nrag loaded")
    except Exception as e:
        print("fail")
        raise e  # noqa: TRY201
    yield

    app_state.clear()


app = FastAPI(title="Pandas API Engine", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Verify if the cached cloud RAG engine is actively mounted in memory."""
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
    evaluator = app_state.get("batch_evaluator")
    if not evaluator:
        print("[WARNING] Batch evaluator instance is not ready.")
        return

    # Call it on the instance variable (now it has 'self' bound correctly)
    results_eval = evaluator.eval()
    eval_logger(results_eval)


@app.post("/api/v1/ask")
async def handle_request(payload: QueryRequest, background_task: BackgroundTasks):
    """
    This endpoint executes fast because it does not load models or files.
    It simply reads the pre-loaded engine from memory.
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
