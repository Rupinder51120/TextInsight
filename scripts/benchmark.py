"""Manual latency benchmark run — docs/LATENCY_AND_PERFORMANCE.md §7. Not a pytest test: this is the
one-off measurement pass whose output feeds directly into README.md's Latency Benchmarks section.
Every number printed here is a real measurement from this run, never invented (CLAUDE.md §5) — copy-paste
the output directly into the README, don't retype from memory.

Run: KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark.py
"""

import statistics
from pathlib import Path

from ingestion import load_csv
from ingestion.store import corpus_store
from tools.evaluate_candidates import evaluate_candidates
from tools.generate_embeddings import generate_embeddings
from tools.profile_dataset import profile_dataset
from tools.semantic_search import semantic_search
from tools.sentiment_analysis import sentiment_analysis
from tools.summarize_text import summarize_text
from tools.text_classification import text_classification

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
N_RUNS = 3


def _load_benchmark_corpus():
    content = (FIXTURES / "csv" / "benchmark_reviews.csv").read_bytes()
    corpus = load_csv(content, "benchmark_reviews.csv")
    corpus_store.put(corpus)
    return corpus


def _report(name: str, cold_ms: float, warm_runs_ms: list[float]) -> None:
    print(f"\n{name}")
    print(f"  cold (first call, includes model load): {cold_ms:.0f} ms")
    print(
        f"  warm (n={len(warm_runs_ms)}): median={statistics.median(warm_runs_ms):.0f} ms, "
        f"range={min(warm_runs_ms):.0f}-{max(warm_runs_ms):.0f} ms"
    )


def main() -> None:
    corpus = _load_benchmark_corpus()
    print(f"Benchmark corpus: {corpus.document_count} documents (tests/fixtures/csv/benchmark_reviews.csv)")

    # --- profile_dataset ---
    times = [profile_dataset(corpus_ref=corpus.corpus_ref).latency_ms for _ in range(N_RUNS)]
    print(
        f"\nprofile_dataset (n={N_RUNS}): median={statistics.median(times):.0f} ms, range={min(times):.0f}-{max(times):.0f} ms"
    )

    # --- sentiment_analysis (batch of 50) ---
    cold = sentiment_analysis(corpus_ref=corpus.corpus_ref).latency_ms
    warm = [sentiment_analysis(corpus_ref=corpus.corpus_ref).latency_ms for _ in range(N_RUNS)]
    _report("sentiment_analysis (batch of 50)", cold, warm)

    # --- text_classification (zero-shot, batch of 50) ---
    labels = ["quality", "shipping", "customer_service", "price"]
    cold = text_classification(corpus_ref=corpus.corpus_ref, candidate_labels=labels).latency_ms
    warm = [
        text_classification(corpus_ref=corpus.corpus_ref, candidate_labels=labels).latency_ms for _ in range(N_RUNS)
    ]
    _report("text_classification (zero-shot, batch of 50)", cold, warm)

    # --- summarize_text (batch_digest over 50 docs) ---
    cold = summarize_text(corpus_ref=corpus.corpus_ref, mode="batch_digest").latency_ms
    warm = [summarize_text(corpus_ref=corpus.corpus_ref, mode="batch_digest").latency_ms for _ in range(N_RUNS)]
    _report("summarize_text (batch_digest, 50 docs)", cold, warm)

    # --- generate_embeddings (index build, cold) vs semantic_search (query, warm) ---
    # A fresh corpus_ref per indexing call to force a genuine rebuild each time (idempotent cache would
    # otherwise report cached:true from the second call on) — indexing latency is one-time-per-corpus by
    # design (docs/TOOLS_AND_MODELS.md #6), so each measurement here uses its own corpus.
    def _fresh_corpus_ref():
        content = (FIXTURES / "csv" / "benchmark_reviews.csv").read_bytes()
        fresh_corpus = load_csv(content, "benchmark_reviews.csv")
        corpus_store.put(fresh_corpus)
        return fresh_corpus.corpus_ref

    cold = generate_embeddings(corpus_ref=_fresh_corpus_ref()).latency_ms
    warm = [generate_embeddings(corpus_ref=_fresh_corpus_ref()).latency_ms for _ in range(N_RUNS)]
    _report("generate_embeddings (index build, 50 docs)", cold, warm)

    generate_embeddings(corpus_ref=corpus.corpus_ref)  # ensure an index exists for the main corpus
    query_times = [
        semantic_search(corpus_ref=corpus.corpus_ref, query="delayed delivery complaints").latency_ms
        for _ in range(N_RUNS)
    ]
    print(
        f"semantic_search (query only, warm, n={N_RUNS}): "
        f"median={statistics.median(query_times):.0f} ms, range={min(query_times):.0f}-{max(query_times):.0f} ms"
    )

    # --- evaluate_candidates (2 candidates x 25 labeled examples) ---
    labeled_content = (FIXTURES / "csv" / "labeled_sentiment.csv").read_bytes()
    labeled_corpus = load_csv(labeled_content, "labeled_sentiment.csv")
    corpus_store.put(labeled_corpus)
    profile = profile_dataset(corpus_ref=labeled_corpus.corpus_ref).model_dump()
    candidate_models = [
        "distilbert-base-uncased-finetuned-sst-2-english",  # already warm from the sentiment_analysis run above
        "cardiffnlp/twitter-roberta-base-sentiment-latest",  # first load happens on this tool's first call
    ]
    cold = evaluate_candidates(
        corpus_ref=labeled_corpus.corpus_ref, profile=profile, candidate_models=candidate_models
    ).latency_ms
    warm = [
        evaluate_candidates(
            corpus_ref=labeled_corpus.corpus_ref, profile=profile, candidate_models=candidate_models
        ).latency_ms
        for _ in range(N_RUNS)
    ]
    _report("evaluate_candidates (2 candidates x 25 labeled examples)", cold, warm)

    print("\nDone. Copy the numbers above into README.md's Latency Benchmarks section verbatim.")


if __name__ == "__main__":
    main()
