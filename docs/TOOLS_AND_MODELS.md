# TOOLS_AND_MODELS.md — TextInsight

Design principle: tools are **cohesive capabilities**, not one-function-per-utility. 12 tools total
(`evaluate_candidates` added to make the model-recommendation feature's "measured on your data" claim real
instead of permanently empty — see `MODEL_RECOMMENDATION.md` §6.5).

## Tool Catalog

### 1. `profile_dataset`
- **Input**: `corpus_ref` (session corpus pointer), optional `column_hint`.
- **Output**: `{ n_documents, text_column, avg_length, length_distribution, detected_language, has_labels,
  label_column?, class_distribution?, source_format }`.
- **Purpose**: mandatory near-first step; parameterizes every downstream tool (which column is text, is this
  large enough to trust class-balance claims, etc.) and feeds `model_recommendation`.
- **Model used**: none (deterministic Pandas/regex/heuristics) + optional fast language-id model
  (`papluca/xlm-roberta-base-language-detection` or `langdetect` library) for `detected_language`.
- **Local/API**: local.
- **Expected latency**: < 1.5s for ≤5k rows.
- **Failure cases**: no text-like column found → returns explicit `text_column: null` and a reason; agent
  must ask the user to clarify rather than guessing silently.
- **Chainable**: yes — output cached in `AgentState.profile`, reused by nearly every other tool/workflow.

### 2. `sentiment_analysis`
- **Input**: `corpus_ref`, `document_ids?` (subset), `text_column` (from profile).
- **Output**: `{ per_document: [{id, label, score}], distribution: {positive, negative, (neutral?)} }`.
- **Purpose**: core sentiment capability; also the standard first step of diagnostic workflows.
- **Model used**: `distilbert-base-uncased-finetuned-sst-2-english` (binary pos/neg, small, fast). Note in UI
  that this model is binary (no neutral class) — an explicit, honest limitation, not hidden.
- **Local/API**: local (HF `pipeline("sentiment-analysis")`).
- **Expected latency**: ~1–3s for a batch of ~50 short texts on CPU.
- **Failure cases**: empty/too-short text rows are skipped with a count reported, not silently dropped.
- **Chainable**: yes — downstream `filter_documents` commonly consumes its output.

### 3. `text_classification`
- **Input**: `corpus_ref`, `document_ids?`, `candidate_labels` (user-provided or inferred from query, e.g.
  billing/technical/delivery/refund).
- **Output**: `{ per_document: [{id, label, score, all_scores}] }`.
- **Purpose**: zero-shot multi-class classification without any training, matching the "classify complaints"
  use case with arbitrary user-defined categories.
- **Model used**: zero-shot NLI model — default `valhalla/distilbart-mnli-12-3` (faster); `facebook/bart-large-mnli`
  offered as an optional higher-accuracy/slower alternative, selectable via config.
- **Local/API**: local (HF `pipeline("zero-shot-classification")`).
- **Expected latency**: ~2–5s for a batch of ~50 short texts with the distil variant on CPU (zero-shot is the
  most latency-sensitive default tool — flagged in `LATENCY_AND_PERFORMANCE.md`).
- **Failure cases**: no `candidate_labels` resolvable from the query or dataset → agent asks a clarifying
  question instead of guessing categories.
- **Chainable**: yes.

### 4. `named_entity_recognition`
- **Input**: `corpus_ref`, `document_ids?`.
- **Output**: `{ per_document: [{id, entities: [{text, type, start, end, score}]}], aggregate_counts }`.
- **Purpose**: "extract organizations/people/locations" use case.
- **Model used**: `dslim/bert-base-NER` (PER/ORG/LOC/MISC, standard, fast enough for CPU on short texts).
- **Local/API**: local (HF `pipeline("ner", aggregation_strategy="simple")`).
- **Expected latency**: ~1–3s for a batch of ~50 short texts.
- **Failure cases**: very long documents truncated to model max length; truncation flagged in output, not
  silent.
- **Chainable**: yes — aggregate counts can feed `synthesize`/`explain_results` directly.

### 5. `summarize_text`
- **Input**: `corpus_ref`, `document_ids?` (single doc or filtered subset), `mode: "single" | "batch_digest"`.
- **Output**: `{ summary: str, source_document_ids: [...] }` (batch mode concatenates/chunks then summarizes
  a representative digest, explicitly labeled as such, not per-document for large batches).
- **Purpose**: document summarization and the final step of diagnostic ("why unhappy") workflows.
- **Model used**: `sshleifer/distilbart-cnn-12-6` default (fast); `facebook/bart-large-cnn` optional
  higher-quality alternative.
- **Local/API**: local.
- **Expected latency**: ~2–6s per single doc / digest chunk of ~1–2k tokens.
- **Failure cases**: input exceeding model context is chunked with map-then-reduce summarization; chunking
  behavior surfaced in output metadata.
- **Chainable**: yes — commonly the last deterministic step before `synthesize`/`explain_results`.

### 6. `generate_embeddings`
- **Input**: `corpus_ref`.
- **Output**: `{ index_id, n_vectors, dim, built: bool, cached: bool }` (idempotent — if a valid index for
  this corpus version already exists, returns `cached: true` without recomputation).
- **Purpose**: builds/refreshes the FAISS index backing semantic search.
- **Model used**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, strong quality/latency tradeoff).
- **Local/API**: local.
- **Expected latency**: a few seconds for ≤5k short texts (one-time per corpus version; see
  `LATENCY_AND_PERFORMANCE.md` for async/background-indexing notes).
- **Failure cases**: corpus too small (e.g., 0 usable texts) → explicit error, no empty index silently
  created.
- **Chainable**: yes — precondition for `semantic_search`.

### 7. `semantic_search`
- **Input**: `corpus_ref`, `query: str`, `top_k`.
- **Output**: `{ results: [{id, text_excerpt, score}] }`.
- **Purpose**: "find complaints about delayed delivery" use case.
- **Model used**: same embedding model as `generate_embeddings` (query-time embedding only — one vector).
- **Local/API**: local; requires an existing FAISS index (auto-triggers `generate_embeddings` if missing,
  agent-level chaining, not hidden inside this tool).
- **Expected latency**: < 300ms typical (single-vector embed + FAISS search).
- **Failure cases**: no index available and building one fails → explicit error, not a silent empty result.
- **Chainable**: yes.

### 8. `filter_documents`
- **Input**: `corpus_ref`, `criteria` (structured — e.g., `{from_tool: "sentiment_analysis", field: "label",
  equals: "negative"}` or `{from_tool: "semantic_search", top_k: 20}`).
- **Output**: `{ document_ids: [...], count }`.
- **Purpose**: the deterministic "glue" step between tools in multi-step workflows (e.g., isolate negative
  reviews before summarizing). Explicitly deterministic — no LLM call — per the project's "prefer
  deterministic code for deterministic tasks" principle.
- **Model used**: none.
- **Local/API**: local.
- **Expected latency**: negligible (< 50ms), in-memory filtering.
- **Failure cases**: criteria references a tool that hasn't run yet → clear error surfaced to `route_next`,
  which should trigger re-planning rather than crash.
- **Chainable**: yes — this is the primary chaining mechanism between analysis tools.

### 9. `model_recommendation`
- **Input**: `profile` (from `profile_dataset`), `task_type`, `user_constraints?` (latency/compute/etc.),
  `research_evidence?` (from `research_models`, optional).
- **Output**: `{ recommendation: str, rationale: [...], evidence: {measured_on_user_data: [...] (usually
  empty — see MODEL_RECOMMENDATION.md), external_research: [...], system_judgment: [...]}, confidence_note }`.
- **Purpose**: turns profile + optional research into the labeled recommendation described in
  `MODEL_RECOMMENDATION.md`. Uses the LLM for the *reasoning/synthesis* step over structured inputs — this
  is one of the two intentional, value-adding LLM uses (the other being `synthesize`).
- **Model used**: LLM (Groq), grounded by rule-based candidate generation (see
  `MODEL_RECOMMENDATION.md` for the rules).
- **Local/API**: hybrid (local rules + Groq API call for the write-up).
- **Expected latency**: dominated by one LLM call, roughly 1–3s.
- **Failure cases**: LLM call fails → falls back to the rule-based candidate list without prose rationale,
  clearly labeled as a degraded response.
- **Chainable**: yes — typically the last step before `synthesize`, or `synthesize` is skipped and this
  tool's output *is* the final answer for pure recommendation queries.

### 10a. `evaluate_candidates` *(the fix for the "we never actually know" gap)*
- **Input**: `corpus_ref`, `profile` (must have `has_labels: true`), `candidate_models` (from
  `model_recommendation`'s rule-based shortlist), `sample_size` (capped, e.g. ≤500).
- **Output**: `{ per_model: [{model_name, accuracy, f1, n_examples}], skipped: bool, skip_reason?: str }`.
- **Purpose**: turns Section A of the recommendation output from a permanent placeholder into a real,
  dataset-specific number — run every shortlisted *pretrained* candidate as inference against a sample of
  the user's own labeled data and score it against ground truth. **No parameters are updated on any model;
  this is inference + scoring, not training**, and stays fully inside the project's "no training" scope.
- **Model used**: whichever candidates the rules in `MODEL_RECOMMENDATION.md` §3 shortlisted (e.g., the
  sentiment shortlist: DistilBERT-SST2, BERT-base-SST2, RoBERTa-base-sentiment).
- **Local/API**: local.
- **Expected latency**: proportional to sample size × number of candidates (e.g., ~5–15s for 3 candidates ×
  200 examples on CPU) — this is the most expensive single step in the recommendation workflow, and is
  skipped entirely (not silently slow-walked) when no usable labels exist.
- **Failure cases**: fewer than a minimum threshold of labeled examples (e.g., <20) → `skipped: true` with
  an explicit `skip_reason`, never a fabricated or extrapolated number.
- **Chainable**: yes — feeds directly into `model_recommendation`'s Section A.

### 10b. `research_models`
- **Input**: `task_type`, `candidate_models` or `topic` (e.g., "sentiment classification benchmark
  DistilBERT vs BERT").
- **Output**: `{ evidence: [{claim, source_title, source_url, snippet}], found: bool }`.
- **Purpose**: live web research backing recommendation claims.
- **Model used**: none directly (search API); optionally the LLM is used to extract/condense a snippet into
  a one-line claim, always retaining the source URL alongside it.
- **Local/API**: API (Tavily or configured alternative) + optional local LLM condensation step.
- **Expected latency**: 1–3s typical for a search API round trip.
- **Failure cases**: API key missing/rate-limited/network failure → `found: false`, agent proceeds with
  system-judgment-only recommendation, explicitly noting research was unavailable.
- **Chainable**: yes — feeds `model_recommendation`.

### 11. `explain_results` (folded into the `synthesize` graph node, exposed as a callable tool for direct/unit
testing)
- **Input**: `tool_results` (arbitrary subset), `user_query`.
- **Output**: `{ explanation: str }`.
- **Purpose**: the final natural-language synthesis over deterministic tool outputs — the one place the LLM
  narrates results, always grounded in structured data passed in (not free invention).
- **Model used**: LLM (Groq).
- **Local/API**: API.
- **Expected latency**: 1–3s, one call.
- **Failure cases**: LLM failure → falls back to a templated, non-LLM summary built directly from
  `tool_results` so the user always gets *something* usable.
- **Chainable**: terminal step — not chained further within a turn.

## Default Pretrained Model Choices (Summary)

| Task | Default model | Why this one (not the biggest) |
|---|---|---|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | Small, fast, purpose-trained for sentiment; adequate accuracy for review-style text. |
| Zero-shot classification | `valhalla/distilbart-mnli-12-3` | Distilled MNLI model; zero-shot is inherently the slowest default tool (one forward pass per label), so latency matters most here. |
| NER | `dslim/bert-base-NER` | Standard, well-benchmarked, base-size (not large), fast on CPU for short texts. |
| Summarization | `sshleifer/distilbart-cnn-12-6` | Distilled CNN/DM summarizer; large BART offered only as an optional slower/better alternative. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Best-known latency/quality tradeoff for semantic search at this scale (384-dim, small). |

All models are loaded once at startup/first-use and reused across requests (see `LATENCY_AND_PERFORMANCE.md`).
