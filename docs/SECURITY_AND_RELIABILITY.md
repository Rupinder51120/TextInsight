# SECURITY_AND_RELIABILITY.md — TextInsight

## 1. File Validation

- File type restricted to `.csv`, `.txt`, `.pdf` by content sniffing (not just extension trust).
- Size limit enforced (`MAX_UPLOAD_MB` config) before any parsing begins.
- CSV parsing uses safe Pandas defaults (no arbitrary code execution paths like pickled objects); PDF
  extraction via PyMuPDF does not execute embedded scripts/JavaScript in the PDF.
- Empty or entirely non-text content (e.g., scanned-image PDF with no extractable text) is rejected with a
  clear error rather than silently producing an empty corpus.

## 2. Prompt Injection Considerations

- **Uploaded document content is treated strictly as data, never as instructions.** Tool functions pass
  document text into model pipelines (HF pipelines) as plain input; the only place document text reaches the
  LLM is inside clearly delimited context blocks in `synthesize`/`model_recommendation` prompts, with system
  instructions that explicitly tell the model to treat that content as data to analyze, not as commands to
  follow.
- **Research/search results are treated the same way** — snippets returned by the search API are external,
  untrusted content; they are quoted narrowly and attributed, never interpreted as instructions to the agent
  (matches the general instruction-source-boundary principle: only the user's actual chat input is a valid
  instruction source).
- Tested explicitly in `TESTING_STRATEGY.md` §10 with adversarial document content containing
  instruction-like text.

## 3. Malicious Documents

- No execution of any embedded content from uploaded files (macros, scripts, embedded objects) — extraction
  libraries used (Pandas CSV reader, PyMuPDF text extraction) do not execute document-embedded code.
- Extremely large or pathological files (e.g., zip-bomb-style CSVs, PDFs with excessive page counts) are
  bounded by the size limit and a page/row count cap, with a clear rejection message.

## 4. API Key Handling

- All API keys (`GROQ_API_KEY`, `TAVILY_API_KEY`, `LANGCHAIN_API_KEY`) loaded from environment variables /
  `.env`, never hardcoded, never logged, never included in any API response body or error message returned
  to the frontend.
- `.env` excluded via `.gitignore`; `.env.example` ships with placeholder values only.

## 5. Size Limits

- Upload size (`MAX_UPLOAD_MB`), row/document count cap for very large CSVs (with a clear "truncated to N
  rows for analysis" notice rather than silent truncation), and PDF page count cap.
- Batch sizes for NLP tools bounded to keep single-request latency and memory bounded (see
  `LATENCY_AND_PERFORMANCE.md` §5).

## 6. Timeouts

- `LLM_TIMEOUT_SECONDS` bounds every Groq call.
- Search API calls bounded by a short timeout (a few seconds) since `research_models` must degrade
  gracefully rather than stall a whole turn.
- Model inference calls, being local and CPU-bound, are bounded implicitly by input size caps rather than a
  wall-clock timeout (a genuinely hung local inference call would indicate a bug, not a transient condition
  worth retrying).

## 7. Retries

- LLM and search API calls use bounded retry (e.g., up to 2–3 attempts) with exponential backoff, only on
  retryable errors (timeouts, 429, 5xx) — never retried indefinitely, and never retried on 4xx
  client-error-type failures (e.g., invalid API key), which surface immediately instead.

## 8. Rate Limits

- Provider-side rate limits (Groq, search API) are respected via the backoff behavior in §7; the project
  does not implement its own user-facing rate limiting for the 5-day scope (single-user/demo context), but
  the `LLMClient`/`ResearchClient` abstraction would be the natural place to add it later.

## 9. Unsafe Tool Execution

- Every tool has a fixed, typed (Pydantic) input schema — the agent cannot invoke a tool with arbitrary
  free-form arguments; malformed tool-call arguments from the LLM fail validation and route to
  `handle_error` rather than executing with bad inputs.
- No tool performs file-system writes outside its designated session directory (e.g., FAISS index files),
  and no tool shells out to arbitrary system commands.
- The system exposes no tool capable of training/fine-tuning, regardless of how a user phrases a request —
  this is enforced by the tool catalog itself (no such tool exists to call), not just by prompting.

## 10. External Research Reliability

- Evidence is source-attributed by construction (`research_models` output schema requires `source_url`
  alongside every claim) — there is no code path that produces an unattributed claim from research.
- If the search API is unavailable, misconfigured, or returns nothing usable, the system explicitly reports
  "no external evidence found/available" rather than falling back to an unattributed LLM guess dressed up as
  research.

## 11. Hallucination Mitigation

- Deterministic NLP outputs (sentiment/classification/NER/summarization/embeddings) come from actual model
  inference, not LLM generation — the LLM only narrates/reasons over these structured results.
- `synthesize`/`explain_results` prompts are constructed to only reference facts present in `tool_results`;
  numeric claims (counts, percentages, scores) are computed in Python and inserted into the prompt/response
  as pre-computed values, not left for the LLM to compute or recall.
- `model_recommendation`'s A/B/C structure (per `MODEL_RECOMMENDATION.md`) is the primary structural defense
  against the specific hallucination risk called out in the project brief (presenting external benchmarks as
  if measured on the user's data).

## 12. Source Attribution

- Every external claim surfaced anywhere in the UI carries a visible source (title + link), consistent
  end-to-end from `research_models`'s output schema through to the Streamlit evidence cards
  (`UI_SPEC.md` §6).
