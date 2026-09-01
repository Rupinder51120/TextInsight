# README_PLAN.md — Planned README.md Structure

This is the outline for the eventual `README.md`, to be filled in as the project is actually built (Day 5).
No section below should contain invented numbers or unimplemented claims when the real README is written.

```markdown
# TextInsight

One-paragraph overview (from RESUME_AND_INTERVIEW.md §2).

## Demo
- Screenshot(s) or short GIF of the Streamlit UI in action (upload → query → multi-step result).
- Link to a hosted demo if one exists; otherwise "run locally" instructions only.

## Architecture
- High-level diagram (reuse the Mermaid diagram from ARCHITECTURE.md §1).
- One paragraph explaining the LangGraph agent (state, nodes, conditional routing) — link to
  ARCHITECTURE.md for full detail.

## Features
- Bulleted feature list (from PROJECT_SPEC.md §5), grouped as NLP Capabilities / Model Guidance / Platform.

## Tech Stack
- Table: technology → role (condensed from TECH_STACK.md).

## Installation
- Python version, `pip install -r requirements.txt` (or `uv`/`poetry` if adopted during build), model
  download note (first run downloads Hugging Face weights).
- **Alternative: Docker.** `docker compose up` builds and starts both the FastAPI backend and Streamlit
  frontend containers; no local Python environment needed. State this as an equally valid path, not a
  fallback — some reviewers will prefer it.

## Configuration
- `.env` setup instructions referencing `.env.example`; list of required vs. optional env vars
  (from API_AND_SERVICES.md §6). Same `.env` file is used whether running locally or via Docker Compose
  (compose reads it automatically; values are never baked into the images).

## Usage
- **Local:**
  - `uvicorn backend.main:app --reload`
  - `streamlit run frontend/app.py`
- **Docker:**
  - `docker compose up --build`
  - Streamlit is reachable at the port mapped in `docker-compose.yml` (state the actual port once decided,
    e.g. `localhost:8501`).
- Basic upload-then-query walkthrough (same regardless of run method).

## Example Queries
- The 10 example use cases from PROJECT_SPEC.md §6, verbatim, each with a one-line note on what workflow it
  triggers.

## Architecture Diagram
- (Same as Architecture section, or a more detailed one — decide during Day 5 polish whether one diagram
  suffices or a second, more detailed sequence diagram from ARCHITECTURE.md §4 adds value.)

## Latency Benchmarks
- Real measured numbers only, in a table, with: what was measured, on what hardware/OS, dataset size used,
  cold vs. warm distinction. Explicitly state "these are local, single-run/median-of-N measurements, not
  formal benchmarks." Populate from LATENCY_AND_PERFORMANCE.md §7 results.

## Limitations
- No model training/fine-tuning is performed — recommendations only (link MODEL_RECOMMENDATION.md).
- Zero-shot classification and multi-step diagnostic workflows are the slowest default flows.
- Research evidence credibility filtering is source-type-based only, not deep fact-checking.
- Session-scoped state only (no persistent multi-session workspace).
- English-first models by default; other-language support not validated.
- Scanned/image PDFs (no extractable text) are not supported (no OCR).

## Future Work
- (From PROJECT_SPEC.md §12) optional local/offline LLM provider, user-supplied labeled eval data for real
  measured model comparisons, persistent workspaces, streaming responses, broader language support.
```

## Notes for Day 5

- Every number in the Latency Benchmarks section must trace back to an actual local run recorded during
  `LATENCY_AND_PERFORMANCE.md` §7's testing pass — copy-paste from test output/logs, don't retype from
  memory.
- The Limitations section should be written honestly and specifically; it is also useful raw material for
  `RESUME_AND_INTERVIEW.md` §9 ("claims we should not make").
