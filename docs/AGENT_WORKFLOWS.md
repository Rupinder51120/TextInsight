# AGENT_WORKFLOWS.md — TextInsight

Each workflow below documents: trigger, relevant state fields, nodes traversed, tool calls, conditions, and
final output shape.

## 1. Sentiment Analysis

- **Trigger**: query intent classified as `sentiment` (e.g., "analyze the sentiment", "how do customers
  feel").
- **State**: `corpus_ref`, `profile` (cached or computed now).
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (sentiment_analysis) → route_next → synthesize`. (*skipped if profile already cached for this corpus.)
- **Tool calls**: `profile_dataset` (conditional), `sentiment_analysis`.
- **Conditions**: none beyond profile-cache check.
- **Final output**: sentiment distribution, representative examples per class, latency breakdown,
  explanation.

## 2. Classification

- **Trigger**: intent `classification` with explicit or inferable candidate labels (e.g., "classify these
  complaints into billing, technical, delivery, refund").
- **State**: `corpus_ref`, `profile`, `candidate_labels` (parsed from query).
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (text_classification) → route_next → synthesize`.
- **Tool calls**: `profile_dataset` (conditional), `text_classification`.
- **Conditions**: if no candidate labels can be parsed from the query and none exist in the dataset, agent
  routes to a clarifying question instead of `text_classification` (short-circuit before tool execution).
- **Final output**: per-document label + confidence, label distribution, explanation.

## 3. NER

- **Trigger**: intent `ner` (e.g., "extract organizations and people").
- **State**: `corpus_ref`, `profile`.
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (named_entity_recognition) → route_next → synthesize`.
- **Tool calls**: `profile_dataset` (conditional), `named_entity_recognition`.
- **Conditions**: none.
- **Final output**: entity list per document, aggregate entity-type counts, explanation.

## 4. Summarization

- **Trigger**: intent `summarization` (e.g., "summarize these documents").
- **State**: `corpus_ref`, `document_ids?` (if user scoped it, e.g., via a prior filter).
- **Nodes**: `understand_intent → plan_steps → execute_tool(summarize_text) → route_next → synthesize`.
- **Tool calls**: `summarize_text` (mode `single` or `batch_digest` depending on document count).
- **Conditions**: document count above a threshold routes to `batch_digest` mode automatically.
- **Final output**: summary text, source document ids, explanation.

## 5. Semantic Search

- **Trigger**: intent `semantic_search` (e.g., "find complaints related to delayed delivery").
- **State**: `corpus_ref`, `query`.
- **Nodes**: `understand_intent → plan_steps → execute_tool(generate_embeddings)* → execute_tool
  (semantic_search) → route_next → synthesize`. (*skipped if a valid cached index exists.)
- **Tool calls**: `generate_embeddings` (conditional/idempotent), `semantic_search`.
- **Conditions**: index staleness check (corpus changed since index built) forces rebuild.
- **Final output**: ranked matches with scores/excerpts, explanation.

## 6. Multi-Step Analysis ("Why are customers unhappy?")

- **Trigger**: intent `diagnostic_explanation` — a "why"/open-ended question requiring synthesis across
  multiple signals.
- **State**: `corpus_ref`, `profile`, accumulating `tool_results`.
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (sentiment_analysis) → route_next → execute_tool(filter_documents: negative) → route_next →
  execute_tool(generate_embeddings)* → execute_tool(semantic_search or grouping) → route_next →
  execute_tool(summarize_text: digest) → route_next → synthesize`.
- **Tool calls**: `profile_dataset`, `sentiment_analysis`, `filter_documents`, `generate_embeddings`
  (conditional), `semantic_search`/topic grouping, `summarize_text`.
- **Conditions**: `route_next` re-plans (bounded) if, e.g., `filter_documents` returns zero negative
  documents — in that case the agent skips straight to a "no negative sentiment detected" synthesis instead
  of continuing a pointless chain.
- **Final output**: causal explanation grounded in filtered/summarized evidence, dominant themes, latency
  breakdown for the whole chain.

## 7. Model Recommendation (dataset-aware, no research)

- **Trigger**: intent `model_recommendation` where the user did not ask for research/benchmarks explicitly,
  or research is unavailable.
- **State**: `corpus_ref`, `profile`, `user_constraints?`.
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (evaluate_candidates)† → execute_tool(model_recommendation) → route_next → synthesize` (synthesize may be
  a thin pass-through here since `model_recommendation`'s output is already prose+structured).
- **Tool calls**: `profile_dataset`, `evaluate_candidates` (†conditional — only runs if `profile.has_labels`
  is true and there are enough labeled examples; see `MODEL_RECOMMENDATION.md` §6.5), `model_recommendation`
  (`research_evidence` omitted).
- **Conditions**: profile-cache check; `evaluate_candidates` is skipped (not run) when labels are absent or
  too few, and that skip reason flows into the final answer rather than being silently dropped.
- **Final output**: labeled recommendation — Section A contains real measured accuracy/F1 numbers if
  `evaluate_candidates` ran, or an explicit skip reason if it didn't; Section B explicitly notes no external
  research was used (per user's request or availability); Section C is the system recommendation.

## 8. Research-Backed Recommendation

- **Trigger**: intent `model_recommendation` where the query implies comparison/benchmarks (e.g., "should I
  use BERT or DistilBERT?", "which model performed best on similar datasets?").
- **State**: `corpus_ref`, `profile`, `research_evidence`.
- **Nodes**: `understand_intent → plan_steps → execute_tool(profile_dataset)* → execute_tool
  (evaluate_candidates)† → execute_tool(research_models) → route_next → execute_tool(model_recommendation) →
  route_next → synthesize`.
- **Tool calls**: `profile_dataset`, `evaluate_candidates` (†conditional, same rule as §7),
  `research_models`, `model_recommendation`.
- **Conditions**: if `research_models` returns `found: false`, `route_next` proceeds to
  `model_recommendation` anyway but with `research_evidence=None`, and the final answer explicitly states
  research was attempted but unavailable — never silently drops the fact that research was requested. The
  `evaluate_candidates` skip/run condition is independent of whether research succeeded — one can be
  populated while the other is empty, and the final answer must be honest about both independently.
- **Final output**: recommendation labeled A (measured on user's data — populated with real numbers whenever
  `evaluate_candidates` ran, per `MODEL_RECOMMENDATION.md` §6.5; explicitly states why if it didn't run), B
  (external research, cited, or explicitly "not found"), C (system judgment).

## 9. Conversational Follow-Up

- **Trigger**: any query in a session where `chat_history` is non-empty and the query references prior
  results ("those negative ones", "now summarize that", "what about entities in the same set").
- **State**: `chat_history`, cached `tool_results` from prior turns.
- **Nodes**: `understand_intent` (resolves references against `chat_history`/prior `tool_results`) →
  `plan_steps` (plan is often shorter — reuses cached results, only executes tools for the *new* part of the
  request) → `execute_tool` (only for new work) → `route_next` → `synthesize`.
- **Tool calls**: only whatever is newly required (e.g., just `summarize_text` if sentiment+filter were
  already done in the prior turn).
- **Conditions**: if a referenced prior result no longer exists (e.g., corpus was replaced), agent explicitly
  says so and asks the user to restate rather than guessing.
- **Final output**: same shape as the relevant single/multi-step workflow above, scoped to only the new
  work performed.
