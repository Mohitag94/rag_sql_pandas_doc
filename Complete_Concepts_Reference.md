# Pandas RAG Project — Complete Concepts & Decisions Reference

**Prepared:** 11 September 2026

Everything covered across the full build: RAG fundamentals, the indexing pipeline, LlamaIndex, SQL, evaluation, deployment, and every deliberate design decision — written so any part of this project can be explained confidently, unprompted.

---

## Part 1 — RAG fundamentals

**What RAG is:** an LLM has never seen your private/specific documents. Ask it directly and it either says "I don't know" or hallucinates a confident-sounding wrong answer. RAG fixes this without touching the model: retrieve the relevant source chunks first, then instruct the LLM to answer using only that context. Two steps — retrieval, then generation.

**RAG is not fine-tuning.** The model's weights never change; only what's fed into the prompt changes, per question. This project's twin, the Grammar Correction project, used fine-tuning (LoRA) — being able to state that contrast clearly is a strong, expected interview answer.

**Indexing happens once; querying happens per request.** Chunking and embedding the corpus (11 pandas doc files) is a one-time cost, done by `build_index.py`. At query time, only the question itself gets embedded — the stored chunk embeddings are reused, never recomputed. This is why RAG stays cheap at query time even as the corpus grows.

**The one legitimate partial analogy to model training:** both have a one-time upfront pass (train a model / build an index) whose result gets reused afterward. What differs: training updates the model's actual weights (learning); indexing runs an already-trained, frozen embedding model purely as a calculator (inference only, no learning). Closer to building a library card catalog than teaching a student.

**Indexing is per-document, then pooled.** Each `.rst` file is chunked and embedded individually; all resulting chunk embeddings — across every file — land in one shared FAISS index, but each chunk retains metadata (`file_name`) pointing back to its source. This is what lets responses cite sources, not just answers.

**Response synthesis ("compact and refine"):** when multiple retrieved chunks are returned, LlamaIndex doesn't necessarily dump them all into one prompt. Its default strategy packs as many chunks as fit into one LLM call for an initial answer, then feeds any leftover chunks back with the existing answer, asking the LLM to refine it if relevant — repeating until all retrieved chunks are processed. This uses the global tokenizer to count how much fits.

---

## Part 2 — Chunking, embedding, tokenization (the concepts most often confused)

**Embedding = vectorization.** Not two separate steps — the same thing. "Embedding" a chunk means: run it through the embedding model, which internally tokenizes it (its own model-specific tokenizer) and outputs a fixed-length vector (list of numbers) representing meaning. Similar meaning → numerically similar vectors.

**Two genuinely separate tokenizers exist in this pipeline, doing different jobs:**
1. **Chunk-splitting tokenizer** — counts tokens to decide where `SentenceSplitter` cuts documents into chunks (`chunk_size` is measured in tokens). Defaults to `tiktoken` unless overridden via `set_global_tokenizer()`.
2. **The embedding model's own internal tokenizer** — converts each chunk's text into that specific model's subword tokens before vectorizing. Fixed, bundled with the model, never configured directly.

**Decision made: match the global (chunk-counting) tokenizer to the embedding model, not the LLM.** Reasoning: embedding models have hard, tight input limits (`bge-small-en-v1.5`: 512 tokens) — undercounting risks silent truncation during embedding. LLMs have much larger context windows (Llama-3.1-8B: 128k tokens), so overflow risk there is far lower. Chosen for consistency and to eliminate the tighter, more easily-violated constraint.

**The global tokenizer is used in two places, not just chunking:** also during query-time response synthesis, for counting how much retrieved context fits in a prompt before hitting the LLM's budget. Initially missed this — `chunk_size`/`chunk_overlap` are indexing-only, but the tokenizer itself is needed in both `build_index.py` and `query_engine.py`.

**`chunk_size = 450`, `chunk_overlap = 50`** — chosen to stay comfortably under `bge-small-en-v1.5`'s 512-token hard limit (tested; 400 and 450 both worked, 450 chosen for slightly less fragmentation). Overlap keeps a slice of context shared between adjacent chunks, avoiding meaning lost right at a chunk boundary.

**The "Token indices sequence length... 6682 > 512" warning is a red herring.** It fires from counting the length of the *entire raw document* before splitting (a generic Hugging Face tokenizer safety warning), not from actual final chunk sizes. Verified by directly tokenizing sample chunks — all landed well under the limit (240–410 tokens). Confirms: always verify against real evidence rather than trusting an alarming-looking warning.

---

## Part 3 — FAISS

**FAISS is a library implementing several index algorithms, not one index type.** This project uses **Flat** (`IndexFlatL2`) — exact search, compares the query against every stored vector, no clustering shortcuts. Correct choice at this scale (few hundred chunks): no accuracy-for-speed tradeoff is needed, so there's no reason to use an approximate method like IVF or HNSW (reserved for million-vector-scale datasets).

**`d = 384`** — the embedding dimensionality, a fixed property of `bge-small-en-v1.5`, not a free choice. Verified empirically (`len(embed_model.get_text_embedding("..."))`, not via a nonexistent `.embed_dim` attribute) rather than assumed.

**FAISS is not automatically used.** `VectorStoreIndex.from_documents()` defaults to LlamaIndex's own in-memory `SimpleVectorStore` unless a `StorageContext` wrapping a real `FaissVectorStore` is explicitly passed in — this was the case in the Week 1 prototype (no FAISS at all) before Week 2 wired it in properly.

**Two separate packages, two separate jobs:** `faiss-cpu` (the real library, `import faiss`, does the actual vector math/storage) and `llama-index-vector-stores-faiss` (a thin adapter, `FaissVectorStore`, letting LlamaIndex's generic `StorageContext` interface talk to a real FAISS index object). Neither replaces the other.

**Persisting the index (`index.storage_context.persist(persist_dir=...)`) matters** because rebuilding (re-embedding all documents) on every run/restart is wasteful. `load_index_from_storage()` reloads a built index from disk without re-embedding anything. `Storage/` is gitignored — fully regenerable from `Data/` + `build_index.py`, so no reason to version it.

---

## Part 4 — LlamaIndex

**What it is:** an orchestration framework, not an embedding/retrieval/generation engine itself — it wires together loaders (`SimpleDirectoryReader`), indexes (`VectorStoreIndex`, talking to FAISS), and query engines, so the "chunk → embed → store → retrieve → generate" pipeline is a handful of calls instead of hand-written glue code.

**Only the original, simpler RAG primitives are used here** — loader → index → query engine — not LlamaIndex's newer agent/Workflows layer (built for multi-step, tool-calling agentic pipelines, out of scope for straightforward document Q&A).

**Alternative considered: LangChain.** Stronger for agent orchestration/tool calling; LlamaIndex is the more retrieval-focused, purpose-built choice for this project's actual need (connecting an LLM to a defined document set).

**`Settings` is a global config object** — set `Settings.embed_model`/`Settings.llm` once, and every subsequent index/query engine call picks them up automatically. Skipping this silently falls back to LlamaIndex's OpenAI defaults, which aren't installed/configured — the root cause of two separate early errors (missing `llama-index-embeddings-openai`, then the same for `llms-openai`).

**LlamaIndex is split into 300+ separate provider packages** — `llama-index-core` has no embedded LLM/embedding code at all; each provider (OpenAI, Hugging Face, Ollama, etc.) is its own installable package, so only what's actually used gets pulled in.

**Query engine, not chat engine — deliberately.** `as_query_engine()` is stateless (one question in, one answer out, no memory of prior questions) — matches the project's one-row-per-question logging design. `as_chat_engine()` (multi-turn, conversation-aware) was never built; would need session/conversation tracking not present in the current schema.

**Documentation quality, honestly assessed:** LlamaIndex's docs are fragmented (300+ near-identical integration pages), inconsistently versioned (old API examples like `ServiceContext` still indexed alongside current `Settings`-based ones), and occasionally nudge toward the paid LlamaCloud product. When in doubt, the actual GitHub source (`llama-index-core`) is more trustworthy than the docs site. Real Python's third-party LlamaIndex tutorial was found to be clearer than the official docs for the loading/indexing/persisting/querying/evaluation five-stage framing — useful as a secondary reference, with the caveat that its examples default to OpenAI, not Hugging Face.

---

## Part 5 — Project architecture and file responsibilities

Each file has exactly one job — established deliberately through iteration, not the first draft:

- **`config.py`** — single source of truth for model names, chunk size, and setup functions (`configure_embedding_and_tokenizer`, `configure_chunking`, `configure_llm`), so a model change is a one-line edit instead of hunting across files.
- **`build_index.py`** (`IndexBuilder`) — one-time: load `.rst` files, chunk, embed (local), build FAISS index, persist to disk. Never touches the LLM.
- **`query_engine.py`** (`IndexLoader` + `QueryEngine`) — `IndexLoader` loads the persisted index (embedding model + tokenizer configured, chunk-size settings not needed here since chunking already happened); `QueryEngine` takes a loaded index, configures the LLM, and answers questions, returning a plain dict (`answer`, `confidence`, `chunk_ids`, `chunk_texts`, `latency_ms`, `error`) — no raw LlamaIndex objects leak out to other files.
- **`db_logger.py`** — the only file that ever opens a Postgres connection, for any table. Named-placeholder (`%(key)s`) inserts accepting plain dicts directly, no manual tuple conversion.
- **`eval_batch.py`** (`BatchEval`) — pure evaluation logic; fetches via `db_logger`, evaluates via `BatchEvalRunner`, returns a list of dicts; does not insert results itself — the caller (`app.py`) does, via `db_logger`, keeping the same "who owns SQL" boundary consistent everywhere.
- **`app.py`** — the conductor. Loads the index and builds `QueryEngine`/`BatchEval` once at startup (via FastAPI's `lifespan`), then on each `/ask` request: calls `QueryEngine.ask()`, and via `BackgroundTasks`, logs the query and conditionally triggers batch evaluation — all after the response has already been sent to the user, so logging/eval cost is invisible to response latency.
- **`streamlit_app.py`** — pure client, calls the deployed FastAPI URL over HTTP; contains no RAG logic at all.

**Why the LLM/embedding config split by file matters:** `build_index.py` needs embedding + tokenizer + chunk size, never the LLM. `query_engine.py`'s `IndexLoader` needs embedding + tokenizer (question embedding, response-synthesis token counting), never chunk size. `QueryEngine` needs the LLM. Each file calls only what it actually needs from `config.py` — verified deliberately rather than assumed symmetrical.

**`.venv` is not branch-specific**, but `pyproject.toml`/`uv.lock` are — installing a new package on one branch doesn't retroactively appear on another; `uv add` (not `uv sync`) is what both installs and records a new dependency.

---

## Part 6 — SQL / database design

**Schema:** all tables under `rag_app`, not the default `public` schema — every query explicitly schema-qualified (`rag_app.query_log`) for unambiguous clarity across files.

**Three tables:**
- `document_metadata` — one row per indexed source file (`source`, `date_added`, `category`)
- `query_log` — one row per `/ask` call (`query_text`, `answer`, `retrieved_chunk_ids`, `retrieved_chunk_texts`, `confidence_score`, `latency_ms`, `error`)
- `eval_results` — one row per evaluated `query_log` row, linked via `query_log_id` **foreign key** (`REFERENCES rag_app.query_log(id)`) — the database itself enforces the relationship, not application code

**Why `%s` placeholders (or named `%(key)s`), never f-strings:** protects against SQL injection — `/ask` accepts arbitrary user text, making this a genuine, not theoretical, security concern.

**Why `retrieved_chunk_ids`/`retrieved_chunk_texts` are `TEXT`, not native arrays:** simplicity at project scale; values are joined with a delimiter before insert, split back into a list on read. IDs use a plain comma (safe — IDs never contain commas); chunk *text* uses a distinct marker (`\n--CHUNK--\n`) since prose can legitimately contain commas/newlines, where a naive comma-join would corrupt the split-back-apart step. Noted limitation: a JSON column or Postgres array type would be more robust; deliberately not used, for simplicity at this scale.

**Why `answer` allows `NULL`:** originally `NOT NULL` with a placeholder string (`"Error Occurred!!!"`) on failure — revised to genuinely allow `NULL`, since `error` already exists specifically to carry failure information; duplicating that signal into `answer` muddies what each column means.

**Eval-fetch filtering:** `fetch_unevaluated()` excludes rows where `error IS NOT NULL` — evaluating a failed query (no real answer generated) is meaningless.

**Batch insert:** `cur.executemany()` for eval results — fewer round-trips than looping individual `INSERT`s; genuinely correct tool at this batch size (15 rows), though not Postgres's fastest possible bulk-insert mechanism (`COPY`, reserved for much larger scale).

---

## Part 7 — Evaluation

**Metrics used: Faithfulness and Relevancy**, via LlamaIndex's built-in evaluation module (`FaithfulnessEvaluator`, `RelevancyEvaluator`) — chosen over adding RAGAS as a separate dependency, since LlamaIndex already ships this and the project is already inside that ecosystem.

**The core evaluation problem:** pandas docs have no pre-made ground-truth Q&A set. Standard fix: generate synthetic test questions from the corpus itself — but this project's actual chosen approach evaluates **real logged user queries** instead of a synthetic set, which is arguably more representative, at the cost of needing chunk text stored in SQL (see Part 6) so eval doesn't need to reload the index.

**Circularity risk, stated honestly:** the same LLM generates answers and judges them — can be lenient grading its own output. Mitigations considered (independent judge model, manual spot-checking) but not implemented at this scale; named as a known limitation.

**Batching:** evaluation runs in batches of 15, triggered when that many un-evaluated rows accumulate — spreads out the extra LLM-judge-call cost (2 extra calls per evaluated question: one for faithfulness, one for relevancy) rather than running on every single live request, which would have tripled per-request LLM usage against a rate-limited free tier.

**`BatchEvalRunner` with `workers=4`** runs evaluations concurrently rather than sequentially — meaningfully faster wall-clock time for a 15-row × 2-metric (30 call) batch, since most time is spent waiting on the HF Inference API, not local compute.

**`nest_asyncio.apply()`** is required because LlamaIndex's evaluators run on an async event loop internally; without patching, this collides with an already-running event loop (e.g. inside FastAPI, itself async) — `nest_asyncio` allows event loops to nest rather than erroring.

---

## Part 8 — API and deployment

**FastAPI is the actual backend** — model loading, retrieval, generation, and SQL logging all happen there. The UI (Streamlit) is a thin client calling it over HTTP; it is not a separate deployment of "the app," just a view onto the already-deployed backend.

**`lifespan` (startup/shutdown hook)** loads the index and builds `QueryEngine`/`BatchEval` exactly once, when the server process starts — stored in a module-level `app_state` dict so every subsequent request reuses the same in-memory objects rather than reconstructing them per request. Only `.ask()` (a method call) repeats per request, not `__init__` (construction).

**`BackgroundTasks`** lets `/ask` return the answer to the user immediately, then continue running logging and (conditionally) batch evaluation afterward, invisible to response latency.

**Deployment memory failure and resolution:** Render's free 512MB tier couldn't hold `torch` + the local embedding model in memory. An attempted fix — moving embeddings to HF's Inference API — failed for a different reason: HF's embedding endpoint (feature-extraction task) has been significantly scaled back as HF shifted focus to its newer Inference Providers router (used for LLM calls), and wasn't reliably serving `bge-small-en-v1.5`. Resolution: reverted to local embeddings everywhere, deployed instead to **Google Cloud Run** (configurable memory, set to 2GB), containerized via a `Dockerfile`. Full platform comparison (Oracle, HF Spaces, Railway, AWS — each considered and rejected for specific reasons) and the complete gcloud setup process are documented separately in `Deployment_and_Additional_Notes.md`.

---

## Part 9 — Deliberately not built (interview-ready, named honestly)

- **RouterQueryEngine combining VectorStoreIndex + SummaryIndex** — routes broad/summarization questions differently from specific factual ones. Rejected: adds a routing LLM call per question; this project's questions are almost entirely factual, not summarization-style.
- **KnowledgeGraphIndex** — entity/relationship extraction and traversal, answering "how does X relate to Y" rather than "what does X do." Mismatched to documentation Q&A's actual question shape.
- **`pgvector` (Neon) instead of FAISS** — would unify vector storage with the rest of the Postgres data in one database. Legitimate alternative; not adopted, to avoid reworking an already-working, tested pipeline.
- **Agentic AI as a separate project** — researched current job-market demand: entry-level AI-skill demand is heavily weighted toward fundamentals (Python/SQL/cloud/visualization with measurable impact); agentic-AI demand is concentrated at mid/senior levels, and shallow "I built an agent" claims are flagged as a red flag under interview scrutiny. Decision: this project already covers the fundamentals well; a genuine (not superficial) agentic project is reasonable *future* polish, not a current gap.
- **Stripping `.rst` markup from chunks before embedding** — the raw pandas `.rst` files contain documentation syntax alongside real prose (e.g. `.. ipython:: python`, `:meth:`, `:ref:` directive tags), observed directly when inspecting a loaded document's raw text. This syntax gets embedded and indexed as-is right now. Not fixed in this build; flagged as a likely retrieval-quality improvement, since the embedding model has to process formatting noise that carries no real semantic meaning.
- **Multimodal RAG** — pandas docs include plots/visualizations (notably in `visualization.rst`) not captured at all, since only text is indexed. Named explicitly as out of scope from the original project plan, not attempted.

---

## Part 10 — Streamlit UI, final deployment, and resume framing

**Streamlit UI implementation:** `st.chat_message` used for a proper chat-style card layout (user/assistant bubbles) rather than plain `st.write` calls; sidebar expander documenting the 11 covered topics for user transparency; `st.session_state.history` capped to the last 5 questions, this-session-only (not shared across users — a deliberate privacy choice, not a limitation, since not every user wants their questions visible to others); a "Clear History" button using `st.rerun()` to force an immediate UI refresh; `timeout=30` on the backend request to avoid an indefinitely hanging call.

**Two real bugs caught during review, one false alarm:**
- **Real bug:** the frontend read `result.get("confidence_score")`, but the backend's `/ask` response actually returns the key `"confidence"` — a silent mismatch (no error thrown, just always `None`) rather than a crash, which is why it went unnoticed until specifically checked.
- **False alarm:** the API URL (`pandas-rag-api-mka-...`) was initially flagged as wrong, assumed to be missing the `-mka` segment based on an earlier, different deployment test. It turned out `-mka` is correct — the actual Cloud Run project is named `rag-pandas-api-mka`, and the service URL reflects that. Lesson: verify against the actual current source of truth (the live deployed URL) rather than an assumption carried over from an earlier step.

**Final deployment, both pieces live:**
- Backend: FastAPI, Dockerized, deployed to **Google Cloud Run**, project `rag-pandas-api-mka`, 2GB memory — `https://pandas-rag-api-mka-576093593374.us-central1.run.app`
- UI: **Streamlit Community Cloud**, deployed straight from the GitHub repo (no separate secrets needed, since `streamlit_app.py` only calls the already-deployed Cloud Run URL over HTTP, holding no credentials itself) — `https://ragsqlpandasdoc-b8vkwfdskpsezxqsmahpe4.streamlit.app/`

**Resume bullet framing, decided:** four ATS-friendly bullets, each leading with a strong action verb, naming concrete tools (LlamaIndex, FAISS, FastAPI, PostgreSQL, Docker, Google Cloud Run, Streamlit) rather than vague description, explicitly naming "pandas documentation" as the grounding domain, and quantifying only what was actually measured (700+ indexed chunks) rather than inventing unmeasured precision (e.g. no fabricated accuracy percentage).

**README additions made after initial draft:** dataset section (source, the exact `curl` collection command used, full 11-file list, and the reasoning for excluding both the other ~25 user-guide files and the auto-generated `reference/` folder), a known-limitation note on `max_tokens` answer cutoffs, a deployment-platforms table, a macOS/Apple Silicon + `uv` development-environment note, and the final live URLs for both the API and the UI.
