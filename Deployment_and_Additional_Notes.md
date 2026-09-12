# Deployment Decisions & Additional Concept Notes

**Prepared:** 11 September 2026

Companion to the main project doc and concepts doc. Covers everything flagged during the build but not yet written up, plus the full deployment story.

---

## 1. Deployment — the real story

**Attempted first:** Render (free tier, 512MB RAM).
**Failed:** local embedding (`sentence-transformers`/`torch`) pushed memory past 512MB. Fix attempted — moving embeddings to Hugging Face's Inference API (`HuggingFaceInferenceAPIEmbedding`) to drop `torch` from the deployed container.
**That fix failed too:** HF's Inference API serves embeddings via the older "feature-extraction" task, which has been significantly scaled back as HF shifted to its newer Inference Providers router (the same router used for LLM chat calls) — `bge-small-en-v1.5` wasn't reliably served this way, returning null results with no clear error.
**Resolution:** reverted to local embeddings everywhere (build and query), and moved to a platform with configurable memory instead of fighting the constraint in code.

**Platforms considered and rejected:**
- Oracle Cloud — free tier exists but setup complexity judged too high for the benefit
- Hugging Face Spaces — free CPU Basic/Docker tier appears to have been recently restricted for new accounts (as of ~July 2026 community reports); also would have repeated the Grammar Correction project's hosting platform, undermining the deliberate platform-diversity goal
- Railway — free tier is the same 512MB ceiling as Render, no actual improvement
- AWS (EC2/App Runner/Lambda) — EC2 free tier still only 1GB RAM; App Runner isn't in AWS's free tier at all; Lambda isn't suited to a long-running app with an in-memory index without re-architecting

**Chosen: Google Cloud Run.** Configurable memory (deployed at 2GB), genuine free tier at this project's traffic scale, containerized via Docker — a more "production-real" deployment pattern than a buildpack PaaS.

### The gcloud setup, conceptually
1. **Authenticate** (`gcloud init`) — links the CLI to a Google account
2. **Create a project** (`gcloud projects create`) — project IDs are globally unique across all GCP users, causing an initial naming collision
3. **Link billing** — a project can exist without billing, but can't deploy anything without it; card required even though usage stays within free tier
4. **Enable three APIs** — Cloud Run (runs the container), Cloud Build (builds the Dockerfile into an image), Artifact Registry (stores the built image)
5. **Fix a default IAM permission gap** — Cloud Build runs as an internal service account, not as the user; new projects don't auto-grant it permission to read uploaded source from storage. Fixed via `roles/cloudbuild.builds.builder` granted to the `PROJECT_NUMBER-compute@developer.gserviceaccount.com` service account
6. **Deploy** (`gcloud run deploy --source .`) — builds and runs in one command, assigns a public HTTPS URL automatically

### Dockerfile — what it is
A plain text file, named exactly `Dockerfile` with no extension, that's a step-by-step recipe for building a container image (base OS + Python + dependencies + code). `docker build` reads it to produce an image (a frozen snapshot); `docker run` starts that image as a live container. Cloud Run's `--source .` flag does both steps automatically.

---

## 2. UI — Streamlit, not Gradio

Doc originally proposed Gradio or Swagger-only. Switched to **Streamlit** instead — better fit for a data-science-adjacent project (more natural for tabular/chart display than Gradio's more form/demo-focused components). FastAPI backend is unchanged either way — Streamlit is a pure client calling the deployed Cloud Run URL via `requests.post()`, same pattern Gradio would have used.

Feature built: last 5 questions + answers + confidence scores, shown via `st.session_state` (this browser session only, not shared across users — deliberately, for privacy, since not everyone wants their questions visible to others).

**Deployed to Streamlit Community Cloud** — connects directly from the GitHub repo, no card or separate secrets required, since `streamlit_app.py` holds no credentials itself (it only calls the already-deployed Cloud Run backend over HTTP).

### Final live URLs
- API (Google Cloud Run, project `rag-pandas-api-mka`): `https://pandas-rag-api-mka-576093593374.us-central1.run.app`
- UI (Streamlit Community Cloud): `https://ragsqlpandasdoc-b8vkwfdskpsezxqsmahpe4.streamlit.app/`

---

## 3. Schema note

All SQL tables live under a dedicated Postgres schema, `rag_app`, not the default `public` schema — e.g. `rag_app.query_log`, `rag_app.document_metadata`, `rag_app.eval_results`. Every query in `db_logger.py` is schema-qualified explicitly.

---

## 4. Evaluation architecture (built)

- **`eval_results` table**, with `query_log_id` as a foreign key back to `query_log` — this relationship itself is the "which rows have been evaluated" tracking mechanism (a `NOT EXISTS` subquery), no separate counter needed.
- **Batch trigger:** evaluation runs once 15 un-evaluated rows (excluding rows where the original query errored) have accumulated, checked via `len(rows) < 15` inside the fetch itself, rather than a separate count query.
- **`BatchEvalRunner`** (LlamaIndex) runs faithfulness + relevancy evaluation concurrently (`workers=4`) using raw strings (`evaluate_response_strs`) — meaning the batch evaluator never needs to reload the FAISS index or reconstruct a query engine, since chunk *text* (not just IDs) is stored directly in `query_log.retrieved_chunk_texts`, delimited with a low-collision-risk marker string rather than a comma.
- **Known limitation, worth stating if asked:** the same LLM generates answers and judges them (faithfulness/relevancy) — a circularity risk, since a model can be more lenient grading its own output. Reasonable at portfolio scale; a stronger design would use an independent judge model.
- Triggered from `app.py` via `BackgroundTasks`, after the response has already been returned to the user — keeps `/ask` latency unaffected by eval cost.

---

## 5. Architecture patterns considered but not built (interview-ready talking points)

- **RouterQueryEngine combining a VectorStoreIndex + SummaryIndex** — would let the system route broad "summarize the docs" questions to a `SummaryIndex` and specific factual questions to the existing vector index. Rejected: adds an extra LLM call per question just for routing, and the pandas-docs use case is overwhelmingly factual/specific, not summarization-style.
- **KnowledgeGraphIndex** — extracts entities/relationships instead of doing similarity search; answers "how does X relate to Y" rather than "what does X do." Rejected as a mismatch for documentation Q&A, which is predominantly factual lookup, not relationship-mapping.
- **Neon's `pgvector` extension as the vector store**, instead of FAISS — would unify vector storage with the rest of the Postgres data (one database instead of FAISS + Neon separately). Rejected in favor of keeping the already-working, already-tested FAISS pipeline; named as a legitimate alternative architecture.
- **RAG's three-category framing** (per a lakeFS blog post): (1) LLMs/platforms with RAG built in (e.g. managed platforms), (2) RAG libraries/frameworks paired with any LLM (LlamaIndex, LangChain), (3) models + separate vector DBs used together. This project sits in categories 2 and 3 combined — LlamaIndex (framework) + FAISS (separate vector DB) + an LLM with no built-in retrieval of its own. Deliberately not Category 1, since assembling the pipeline is the actual skill being demonstrated.
- **Other RAG eval tooling surveyed:** RAGAS, DeepEval, TruLens, LangSmith, Langfuse, Arize Phoenix — mostly aimed at production/enterprise monitoring, more machinery than this project's scale needs. LlamaIndex's own built-in evaluation module (used here) was the simplest fit, no extra dependency.

---

## 6. Small but real fixes along the way (good "one thing that broke and how you fixed it" interview material)

- Token/tab indentation mismatch (editor set to spaces, pasted code used tabs → `TabError`) — fixed via editor settings + "Convert Indentation to Tabs."
- `.env` accidentally wiped by `echo "..." > .env` (overwrite) instead of `>>` (append) — lost `HF_TOKEN` temporarily.
- HF Inference API 403 error — token lacked the "Inference Providers" permission scope; fixed by creating a fine-grained token with that permission explicitly enabled.
- Settings/config duplicated across multiple files (embedding model name, chunk size, etc.) — refactored into a single shared `config.py` as the source of truth.
