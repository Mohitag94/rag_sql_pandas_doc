# Pandas Docs Q&A — RAG + SQL Monitoring + Evaluation

A retrieval-augmented generation (RAG) system that answers natural-language questions about pandas, grounded in the official pandas documentation — built to demonstrate an end-to-end ML engineering pipeline: retrieval, deployment, SQL-based logging, and automated LLM evaluation, not just a notebook accuracy score.

**Live demo:** [Streamlit UI](https://ragsqlpandasdoc-b8vkwfdskpsezxqsmahpe4.streamlit.app/)
**API:** FastAPI backend on Google Cloud Run — `https://pandas-rag-api-mka-576093593374.us-central1.run.app`
**API docs:** [`/docs`](https://pandas-rag-api-mka-576093593374.us-central1.run.app/docs) (Swagger, auto-generated)

**Development environment:** built end-to-end on macOS (Apple Silicon), using `uv` for Python environment/package management.

## Problem

Finding the right pandas method or usage pattern often means scrolling through pages of documentation. This system lets a user ask a plain-English question and get a direct, source-grounded answer instead.

## Architecture

```
Pandas docs (.rst, curated 11-file subset of the user guide)
        |
        v
Chunk + Embed (sentence-transformers, local) --> FAISS index (persisted to disk)
        |
   query time
        |
   Embed question --> FAISS top-k retrieval --> LLM generates answer (Llama-3.1-8B via HF Inference API)
        |
        v
Log to Postgres (Neon): query, retrieved chunks, confidence, latency, error
        |
        v
Batch evaluation (every 15 logged queries): faithfulness + relevancy, via LlamaIndex's evaluation module
        |
        v
Streamlit UI <-- FastAPI backend (Docker, Google Cloud Run)
```

## Stack

| Component | Tool |
|---|---|
| RAG framework | LlamaIndex |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`), local |
| Vector store | FAISS |
| LLM | Llama-3.1-8B-Instruct via Hugging Face Inference API |
| Database | PostgreSQL (Neon), schema `rag_app` |
| Backend | FastAPI |
| Evaluation | LlamaIndex evaluation module (Faithfulness, Relevancy), batched |
| Deployment | Docker + Google Cloud Run |
| UI | Streamlit |
| Environment/packages | uv |

## Project structure

```
config.py           # shared model names, chunk size, LLM/embedding setup — single source of truth
build_index.py      # IndexBuilder — one-time: load docs, chunk, embed, persist FAISS index
query_engine.py     # IndexLoader + QueryEngine — load persisted index, answer questions
db_logger.py        # all Postgres access: query_log, eval_results, document_metadata
eval_batch.py       # BatchEvaluator — faithfulness/relevancy eval on un-evaluated logged queries
app.py              # FastAPI: /ask, /health — orchestrates the above, background logging + eval
app_ui.py           # UI, calls the deployed FastAPI backend
Dockerfile
Data/               # curated pandas .rst files
Storage/            # persisted FAISS index (gitignored, regenerable)
```

## Dataset

**Source:** official pandas GitHub repository, `doc/source/user_guide/` — https://github.com/pandas-dev/pandas/tree/main/doc/source/user_guide

**Collection method:** downloaded directly from GitHub's raw content URLs via `curl` (no `git clone` — avoids pulling pandas' full ~150–250MB codebase for 11 documentation files).

```bash
mkdir -p Data
cd Data
for f in 10min basics indexing merging groupby missing_data reshaping text timeseries io visualization; do
  curl -L "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/source/user_guide/${f}.rst" -o "${f}.rst"
done
cd ..
```

**Files included (11 of the user guide's ~35–40 total):**
- `10min.rst` — 10-minute intro / quick start
- `basics.rst` — essential functionality
- `indexing.rst` — indexing and selecting data
- `merging.rst` — merge, join, concatenate, compare
- `groupby.rst` — split-apply-combine operations
- `missing_data.rst` — working with missing/NaN data
- `reshaping.rst` — reshaping and pivot tables
- `text.rst` — working with text/string columns
- `timeseries.rst` — time series and date functionality
- `io.rst` — reading/writing data (CSV, Excel, etc.)
- `visualization.rst` — plotting and graphics

**Why only these 11, not the full user guide:** the remaining ~25 files cover fairly niche topics (extension arrays, sparse data, styling, plotting backends, etc.) that a typical user is unlikely to ask about, and including them would risk answers the maintainer can't confidently verify. The `reference/` folder (auto-generated API stubs, thousands of files) was excluded entirely — it's function-signature text, not explanatory prose, and isn't useful for chunked retrieval. The goal was a corpus broad enough for real coverage while staying small enough to hold in one person's head for verification and stay fast/cheap to index.

## Known limitation — answer length

Generated answers are capped by a `max_tokens` setting on the LLM (currently 512). Questions requiring long explanations or extensive code generation may occasionally be cut off mid-sentence or mid-code-block. This is a deliberate speed/cost tradeoff, not a retrieval failure — surfaced as a warning directly in the Streamlit UI.

## Deployment platforms

| Component | Platform | Notes |
|---|---|---|
| Backend (FastAPI) | **Google Cloud Run** | Containerized via Docker; deployed at 2GB memory |
| UI (Streamlit) | **Streamlit Community Cloud** | Deploys from GitHub, calls the Cloud Run backend over HTTP |
| Database | **Neon** (managed PostgreSQL) | Free tier, schema `rag_app` |
| LLM + embeddings | **Hugging Face Inference API / local `sentence-transformers`** | LLM generation via HF Inference API; embeddings run locally |

**Note:** the backend was originally attempted on **Render**, but its 512MB free-tier memory limit couldn't hold the local embedding model in memory. Full story — including a failed attempt to move embeddings to HF's Inference API, and why Cloud Run was chosen over Oracle Cloud, Hugging Face Spaces, Railway, and AWS — is documented in `Deployment_and_Additional_Notes.md`.

## Running locally

```bash
uv sync
uv run python build_index.py       # one-time: build the index
uv run uvicorn app:app --reload    # backend, http://localhost:8000
uv run streamlit run streamlit_app.py   # UI, http://localhost:8501
```

Requires a `.env` with `HF_TOKEN` (Hugging Face access token, Inference Providers permission enabled) and `DATABASE_URL` (Neon Postgres connection string).

## Deployment

Backend is containerized (`Dockerfile`) and deployed to Google Cloud Run under project `rag-pandas-api-mka` (2GB memory — see `Deployment_and_Additional_Notes.md` for why Render's free tier wasn't viable for this workload, and the full platform-selection reasoning).

```bash
gcloud run deploy pandas-rag-api-mka --source . --region us-central1 --memory 2Gi \
  --allow-unauthenticated \
  --set-env-vars HF_TOKEN=...,DATABASE_URL=...
```

## What I'd improve next

- Independent judge model for evaluation, rather than reusing the same LLM that generates answers
- `pgvector` (Neon) as an alternative to FAISS, unifying vector storage with the rest of the Postgres data
- Strip `.rst` markup syntax from chunks before embedding, rather than leaving directive/reference syntax in the indexed text
- Multimodal RAG (out of scope for this build)

## Notes

- Render's free tier (512MB) couldn't hold the local embedding model in memory — documented in `Deployment_and_Additional_Notes.md` along with the full resolution path.
- SQL is used for both operational logging (`query_log`) and monitoring/evaluation (`eval_results`) — a deliberately different SQL usage pattern from purely analytical SQL.