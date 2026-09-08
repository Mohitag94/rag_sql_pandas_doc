from contextlib import asynccontextmanager

from fastapi import FastAPI

from query_engine import IndexLoader, QueryEngine

# a plain dict to hold shared state across requests
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts. Loads the persisted index and
    builds the query engine, so every request reuses the same objects
    instead of reloading them per request."""
    loader = IndexLoader()
    app_state["engine"] = QueryEngine(loader.index)
    yield
    # (anything after yield runs once on shutdown — nothing needed here yet)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(question: str):
    engine = app_state["engine"]
    result = engine.ask(question)
    return result
