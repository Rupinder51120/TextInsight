"""evaluate_candidates real-model correctness — docs/TESTING_STRATEGY.md §7: "a small fixture with known
labels and an obviously-better-vs-worse pair of candidate models asserts the tool reports higher accuracy
for the model that should actually perform better on that fixture (sanity check, not a real benchmark
claim)."

The "obviously worse" model is a real sentiment model whose output labels never match our normalization
vocabulary (see tools/evaluate_candidates.py's docstring on scope-limited label normalization) — a stand-in
for "a candidate that clearly performs worse on this scoring," which is the honest, reproducible way to
build this sanity check without depending on one specific alternate model's live accuracy on a given day.
"""

from pathlib import Path

from ingestion import load_csv
from ingestion.store import corpus_store
from tools.evaluate_candidates import evaluate_candidates
from tools.profile_dataset import profile_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def _labeled_corpus_ref() -> str:
    content = (FIXTURES / "csv" / "labeled_sentiment.csv").read_bytes()
    corpus = load_csv(content, "labeled_sentiment.csv")
    corpus_store.put(corpus)
    return corpus.corpus_ref


class TestEvaluateCandidatesRealModels:
    def test_default_sentiment_model_scores_highly_on_clean_labeled_fixture(self):
        corpus_ref = _labeled_corpus_ref()
        profile = profile_dataset(corpus_ref=corpus_ref).model_dump()
        assert profile["has_labels"] is True
        assert profile["label_column"] == "label"

        result = evaluate_candidates(
            corpus_ref=corpus_ref,
            profile=profile,
            candidate_models=["distilbert-base-uncased-finetuned-sst-2-english"],
        )

        assert result.skipped is False
        assert result.per_model[0].n_examples == 25
        # DistilBERT-SST2 on unambiguous, clearly-worded positive/negative reviews should score highly —
        # sanity check per TESTING_STRATEGY §7, not a rigorous benchmark claim.
        assert result.per_model[0].accuracy >= 0.8

    def test_better_vs_worse_candidate_pair_ranks_correctly(self):
        # The "worse" candidate here is textattack/bert-base-uncased-SST-2, verified live (2026-09-01) to
        # output raw, unmapped "LABEL_0"/"LABEL_1" rather than "positive"/"negative" — it was deliberately
        # NOT added to model_recommendation's shortlist for exactly this reason (see
        # tools/model_recommendation.py's _TASK_CANDIDATES comment). It's used here only as a reproducible
        # "obviously worse under this tool's scoring" fixture, independent of which specific alternate
        # model happens to be live-best on any given day — the real assertion under test is that
        # evaluate_candidates correctly differentiates and ranks two real candidates, not a claim about
        # either model's real-world quality.
        corpus_ref = _labeled_corpus_ref()
        profile = profile_dataset(corpus_ref=corpus_ref).model_dump()

        result = evaluate_candidates(
            corpus_ref=corpus_ref,
            profile=profile,
            candidate_models=[
                "distilbert-base-uncased-finetuned-sst-2-english",
                "textattack/bert-base-uncased-SST-2",
            ],
        )

        by_name = {m.model_name: m for m in result.per_model}
        good = by_name["distilbert-base-uncased-finetuned-sst-2-english"]
        mismatched = by_name["textattack/bert-base-uncased-SST-2"]

        assert good.accuracy >= 0.8
        assert mismatched.accuracy < good.accuracy
        assert mismatched.accuracy == 0.0  # raw LABEL_0/LABEL_1 never matches positive/negative ground truth

    def test_second_shortlisted_candidate_has_clean_labels_and_scores_reasonably(self):
        # cardiffnlp/twitter-roberta-base-sentiment-latest IS the shortlist's actual alternative (see
        # tools/model_recommendation.py) — verified live to output clean positive/negative/neutral labels,
        # so it should score meaningfully (not necessarily as high as the SST2-specific default) rather
        # than collapsing to 0 the way a label-mismatched model would.
        corpus_ref = _labeled_corpus_ref()
        profile = profile_dataset(corpus_ref=corpus_ref).model_dump()

        result = evaluate_candidates(
            corpus_ref=corpus_ref,
            profile=profile,
            candidate_models=["cardiffnlp/twitter-roberta-base-sentiment-latest"],
        )

        assert result.per_model[0].accuracy > 0.5
