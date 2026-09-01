"""Real-model NLP output validation — docs/TESTING_STRATEGY.md §4.

Unlike the *_logic.py test files, these run the actual default models from docs/TOOLS_AND_MODELS.md
(first run downloads them, which is slow — this is exactly why §1 keeps them separate from the mocked
logic tests). No exact-score/exact-text assertions per §4 ("scores can shift slightly across model
versions" / "summaries are non-deterministic") — only label-direction, top-label, and shape/length
sanity checks.
"""

from ingestion.corpus import Document
from tests.helpers import register_corpus
from tools.sentiment_analysis import sentiment_analysis
from tools.summarize_text import summarize_text
from tools.text_classification import text_classification


class TestSentimentRealModel:
    def test_label_direction_on_known_examples(self):
        docs = [
            Document(
                id="pos",
                text="I absolutely love this product! Best purchase I've made all year, highly recommend it.",
            ),
            Document(
                id="neg",
                text="Absolutely terrible. It broke immediately and customer service was rude and unhelpful.",
            ),
        ]
        ref = register_corpus(docs)

        result = sentiment_analysis(corpus_ref=ref)

        by_id = {r.id: r.label for r in result.per_document}
        assert by_id["pos"] == "positive"
        assert by_id["neg"] == "negative"
        assert result.latency_ms > 0


class TestTextClassificationRealModel:
    def test_top_label_correctness_on_unambiguous_example(self):
        docs = [Document(id="0", text="My card was charged twice for the same order.")]
        ref = register_corpus(docs)

        result = text_classification(
            corpus_ref=ref,
            candidate_labels=["billing", "technical", "delivery", "refund"],
        )

        assert result.per_document[0].label == "billing"
        assert result.latency_ms > 0


class TestSummarizeTextRealModel:
    def test_summary_is_non_empty_and_shorter_than_source(self):
        source = (
            "The customer service team responded quickly to the initial complaint about the "
            "damaged package. They offered a full refund and apologized for the inconvenience "
            "caused by the delayed shipment. The replacement item arrived within three business "
            "days and matched the original order description exactly. Overall the resolution "
            "process was handled professionally despite the rocky start, and the customer later "
            "left a follow-up note thanking the support agent by name for the quick turnaround."
        )
        docs = [Document(id="0", text=source)]
        ref = register_corpus(docs)

        result = summarize_text(corpus_ref=ref, document_ids=["0"], mode="single")

        assert result.summary.strip() != ""
        assert len(result.summary) < len(source)
        assert result.latency_ms > 0
