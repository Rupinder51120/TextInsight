# UI_SPEC.md — TextInsight (Streamlit)

## 1. Page Layout

Single-page Streamlit app with a two-column layout:

- **Left sidebar**: file upload, session info (corpus name, row/doc count once profiled), settings
  (candidate labels for classification if the user wants to pre-supply them, toggle for enabling research).
- **Main area, top to bottom**:
  1. Chat-style query input (`st.chat_input`), with prior turns rendered above it (`st.chat_message`).
  2. **Agent workflow/status panel** — for the turn currently processing or most recently completed, shows
     the ordered list of nodes/tools executed (e.g., "profile_dataset ✓ → sentiment_analysis ✓ →
     filter_documents ✓ → summarize_text ✓ → synthesize ✓"), each with a small latency tag.
  3. **Result panel** — renders per response type (see §4).
  4. **Latency panel** — collapsible section with the full per-step timing breakdown (table + small bar
     chart via `st.bar_chart`).

## 2. Upload Flow

- `st.file_uploader` accepting `.csv`, `.txt`, `.pdf`, single file at a time (multi-file explicitly
  out of scope for MVP).
- On upload: immediate call to backend `/upload`; on success, show a short profiling summary card (row/doc
  count, detected text column, detected language, label presence) so the user has confirmation before
  querying.
- On failure (bad format, too large, unreadable): clear inline error, no silent fallback.
- Uploading a new file mid-session shows a confirmation that this replaces the active corpus (per
  `DATA_FLOW.md` §3).

## 3. Query Input

- `st.chat_input` placeholder text with example prompts (rotating or static list drawn from
  `PROJECT_SPEC.md` §6 use cases) to guide first-time users.
- Submitted queries are sent to backend `/query` with `session_id`; the UI shows a spinner/status
  (`st.status`) that updates live as each workflow step completes (backend can stream step updates via
  Server-Sent Events or the UI can poll — implementation detail decided in `FIVE_DAY_BUILD_PLAN.md`; simple
  polling is the MVP-safe choice).

## 4. Result Cards (per response type)

- **Sentiment**: distribution bar/pie chart (`st.bar_chart` or Plotly), a small table of example documents
  per class, explanation text.
- **Classification**: label distribution chart, per-document table (text excerpt, predicted label,
  confidence), explanation text.
- **NER**: aggregate entity-type counts (bar chart), expandable per-document entity highlighting (colored
  spans via a small HTML/markdown render), explanation text.
- **Summarization**: summary text in a highlighted card, list of source document ids/excerpts it was drawn
  from, explanation text.
- **Semantic search**: ranked results table (excerpt, similarity score), explanation text.
- **Multi-step diagnostic**: the workflow status panel (§1.2) is especially important here — show each
  intermediate result (sentiment distribution → filtered count → theme summary) as expandable sub-sections,
  culminating in the final explanation.
- **Model recommendation**: three clearly separated, visually distinct sections matching
  `MODEL_RECOMMENDATION.md` §7 (A: measured on your data, B: external research with clickable source links,
  C: system recommendation), plus the confidence/uncertainty note.

## 5. Charts

- Prefer simple built-in Streamlit charts (`st.bar_chart`, `st.line_chart`) for distributions; use Plotly
  (`st.plotly_chart`) only where richer interactivity (hover tooltips on entity counts, similarity scores)
  meaningfully helps — not decoratively.

## 6. Research Evidence Display

- Each evidence item rendered as a small card: claim (LLM-condensed, one line), source title as a clickable
  link, and a "source" badge indicating type (paper/model card/benchmark/other) if determinable.
- An explicit "No external evidence found" state, visually distinct (muted, not an error), when
  `research_models` returns `found: false`.

## 7. Latency Display

- Always visible (collapsed by default) per response: total time, and an expandable per-step table/chart.
- A small "cold start" badge on the first request after app launch, since first-call latency includes model
  loading (per `LATENCY_AND_PERFORMANCE.md` §3) — avoids the UI implying steady-state speed on turn one.

## 8. Errors

- Ingestion errors: inline red `st.error` near the uploader, with the specific reason (unsupported format,
  file too large, empty/unreadable content).
- Tool/agent errors: rendered as a distinct "partial result" state when some tools succeeded and one failed
  (show what did work, then a clear note on what didn't and why), rather than a blank failure.
- LLM/provider errors: a clear "the assistant's reasoning step failed, here are the raw results" fallback,
  consistent with the degraded-mode behavior in `ARCHITECTURE.md` §11 and `API_AND_SERVICES.md`.

## 9. Loading States

- `st.status`/spinner during processing, updating text as each agent step completes (not just a generic
  spinner the whole time) — reinforces the "this is actually agentic, multi-step" story visually.
- Upload processing (profiling) has its own shorter loading indicator, separate from query processing.

## 10. Conversation History

- Rendered as a standard chat thread (`st.chat_message` for user/assistant turns) above the input box.
- Each assistant turn in history remains expandable to re-view that turn's workflow/latency panel (not just
  the final text), so the user can scroll back and inspect exactly what ran for a prior question.
- A "clear session" control resets `chat_history`, cached `tool_results`, and (optionally) the corpus.
