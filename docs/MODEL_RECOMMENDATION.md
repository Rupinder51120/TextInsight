# MODEL_RECOMMENDATION.md — TextInsight

This is the most reputation-sensitive feature in the project: it must never overstate what is known. This
document defines exactly how a recommendation is built so the implementation cannot accidentally blur
evidence sources.

## 1. Dataset Characteristics Extracted (from `profile_dataset`)

- `n_documents` — dataset size.
- `text_length` — average / distribution (short reviews vs long documents materially change model choice).
- `n_classes` / `class_distribution` — if labels are present (classification/sentiment tasks).
- `has_labels` — whether any ground truth exists at all (drives the fine-tune-vs-pretrained answer).
- `detected_language`.
- `domain` — best-effort inferred from column names/content sample (e.g., "product reviews", "support
  tickets") via a lightweight LLM classification of a small text sample, clearly labeled as inferred, not
  asserted as fact.
- `source_format` — CSV/TXT/PDF (affects expected document length/structure).

## 2. User Constraints Extracted

- Parsed from the query and, when ambiguous, asked for explicitly (not guessed):
  - `latency_requirement` (e.g., "fast", "real-time", or unspecified → assume "no strict requirement" and
    say so).
  - `compute_constraints` (e.g., "CPU only", "no GPU available" — default assumption is CPU-only unless
    stated otherwise, matching the project's own deployment target).
  - `task_type` (sentiment / classification / NER / summarization / embeddings — usually inferable from the
    query itself).

## 3. Candidate Model Generation

Deterministic, rule-based, **not** LLM-invented:

1. Start from the fixed per-task candidate shortlist maintained in `TOOLS_AND_MODELS.md`'s default plus 1–2
   well-known named alternatives per task (e.g., for sentiment: DistilBERT-SST2 default, BERT-base-SST2 and
   RoBERTa-base sentiment variants as named alternatives commonly asked about by users).
2. Filter/rank candidates against extracted characteristics using simple, explainable rules, e.g.:
   - Small dataset + no labels → favor zero-shot/pretrained-as-is over anything implying fine-tuning.
   - Strict latency requirement or CPU-only → favor distilled/smaller variants.
   - Larger, labeled dataset with lax latency needs → note that fine-tuning *could* be worth exploring
     (recommendation only — never executed, per scope).
3. The rule engine outputs a short ranked candidate list with the *reason* per candidate — this is the
   deterministic backbone the LLM prose is required to stay consistent with (the LLM is prompted with this
   structured output and instructed to explain it, not to invent its own ranking).

## 4. External Research Retrieval

- `research_models` is called with the task type + candidate model names + inferred domain.
- Returns a small set of attributed evidence items: `{claim, source_title, source_url, snippet}`.
- Evidence is filtered to sources that are plausibly credible by type (papers, official model cards,
  official benchmark pages, recognized technical publications) — the tool does not attempt deep credibility
  scoring beyond source-type filtering, and this limitation is stated in `LIMITATIONS` sections (README,
  resume doc) rather than overclaimed.

## 5. Evidence Ranking

Simple, explainable heuristic (no ML ranking model — keeps this auditable):
1. Prefer evidence explicitly mentioning the same task type (e.g., "sentiment classification") over generic
   mentions of a model.
2. Prefer evidence mentioning a comparable domain (e.g., "reviews", "social media text") when the dataset's
   inferred domain is known.
3. Prefer official model cards/paper abstracts over secondary blog summaries when both are available.
4. Cap the number of evidence items surfaced (e.g., top 3–5) to keep the answer readable and avoid
   evidence-dumping.

## 6. Recommendation Generation

The LLM receives, as structured context:
- The rule-based ranked candidate list + reasons (from §3).
- The ranked evidence list (from §5), or an explicit "no evidence found" flag.
- The extracted profile + constraints (from §1–2).

The LLM is instructed (system prompt, enforced by prompt design and spot-checked in
`TESTING_STRATEGY.md`) to:
- Never state a model is objectively "best" unless something was actually evaluated on the user's own data
  (which, per current scope, is essentially never — see §7).
- Phrase the recommendation as **"Recommended based on dataset characteristics and stated constraints"**
  (or equivalent), matching the exact framing required by the project brief.
- Cite every external claim with its source (title/URL) inline or in a clearly delimited evidence list.
- Explicitly state when no external evidence was found or research wasn't requested.

## 6.5 Measured Evaluation on the User's Own Data (No Training)

This is the fix for the original design's biggest weakness: Section A below was permanently empty, so every
recommendation reduced to "external research + a guess." That's no longer true whenever the dataset already
has labels.

- **Trigger**: `profile_dataset` reports `has_labels: true` (e.g., a sentiment CSV with a `label` column, a
  ticket dataset with a `category` column).
- **What happens**: a new tool, `evaluate_candidates` (see `TOOLS_AND_MODELS.md`), runs each shortlisted
  *pretrained* candidate model (from §3's rule-based shortlist) directly against a held-out sample of the
  user's own labeled data — **inference only, zero parameter updates, zero training loop** — and computes
  real accuracy/F1 on that sample.
- **Why this stays in scope**: nothing is trained or fine-tuned; every candidate is used exactly as
  published. This is evaluation, not training — the same category of operation as `sentiment_analysis`
  itself, just run for several candidate models and scored against ground truth instead of just reported.
- **Sample size discipline**: capped (e.g., ≤500 labeled rows) to keep this fast and CPU-tractable; the
  actual sample size used is always reported alongside the numbers so nobody mistakes a 50-row sample for a
  rigorous benchmark.
- **Failure/skip case**: no labels present, or labels too few to be meaningful (e.g., <20 examples) →
  `evaluate_candidates` is skipped entirely and Section A explicitly says why (see §7), rather than silently
  omitted.

## 7. How External Benchmarks Are Separated From the User's Own Results — Output Contract

Every `model_recommendation` response is structured into three explicitly labeled sections, always present
(even if a section is empty, it is empty *and labeled as such*, never omitted silently):

```
A. Results measured on your dataset:
   - If evaluate_candidates ran: real accuracy/F1 per candidate model, on N held-out labeled examples
     (N always stated). This is now the system's strongest, most defensible claim — an actual number,
     not a guess.
   - If it didn't run (no labels, or too few): explicit statement of which condition applied
     ("Your dataset has no labels, so no evaluation could be run" / "Only N labeled examples were found —
     too few for a meaningful comparison"), never silently omitted.

B. Results reported by external research:
   - [claim] — [source title] ([source URL])
   - ... (or: "No external evidence was found/retrieved for this query.")

C. System recommendation:
   - [Recommended model/approach], based on: [dataset characteristics + constraints], with explicit
     uncertainty language where relevant (e.g., "with your current dataset size, X is a reasonable default;
     this has not been benchmark-verified specifically on your data").
```

This exact separation is a hard requirement of the tool's output schema (Pydantic — `measured_on_user_data:
list`, `external_research: list`, `system_judgment: str`), not just a prompting convention, so the structure
survives even if the LLM's prose drifts.

## 8. Communicating Uncertainty

- Every recommendation includes a one-line confidence/uncertainty note derived from concrete signals: dataset
  size (small datasets → lower confidence language), evidence availability (no research found → lower
  confidence, explicitly flagged), and domain-match quality of any evidence used.
- The system never uses unqualified superlatives ("best", "state-of-the-art") without immediately attaching
  the qualifying source or scope of that claim.

## 9. Fine-Tune vs. Pretrained Advisory

- Governed entirely by rules in §3 plus explicit scope language: the response may say fine-tuning "would
  likely help" or "is unlikely to be necessary" based on dataset size/label availability/latency needs, but
  must always include a sentence equivalent to: "This system does not perform training or fine-tuning; this
  is guidance only." This sentence is templated (not left to the LLM to remember to say), guaranteeing scope
  compliance regardless of prompt drift.
- When §6.5's evaluation ran, the fine-tune-vs-pretrained framing gets sharper for free: if the
  best pretrained candidate's measured accuracy is already high on the user's own data, that's a real,
  specific reason to say fine-tuning is unlikely to be worth it — not just a generic heuristic.
