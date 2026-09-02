"""Streamlit frontend tests via streamlit.testing.v1.AppTest — exercises the actual app script and its
rendering functions without a browser (no browser automation available in this environment; AppTest does
not support simulating st.file_uploader interaction, so these tests pre-seed st.session_state as if an
upload already completed, which is how every response-rendering code path is actually reached).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "frontend" / "app.py")

_CORPUS_INFO = {
    "session_id": "t",
    "corpus_ref": "t",
    "source_filename": "reviews.csv",
    "source_format": "csv",
    "document_count": 5,
    "truncated": False,
    "profile": {"text_column": "review_text", "detected_language": "en", "has_labels": False},
}


def _app_with_history(*turns: dict) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.session_state["corpus_info"] = _CORPUS_INFO
    at.session_state["session_id"] = "t"
    at.session_state["chat_history"] = list(turns)
    at.session_state["first_query_done"] = True
    at.run(timeout=30)
    return at


class TestInitialLoad:
    def test_no_exceptions_before_any_upload(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        assert not at.exception
        assert at.title[0].value == "TextInsight"
        assert any("Upload a CSV" in i.value for i in at.info)


class TestResultRendering:
    def test_all_tool_result_types_render_without_exception(self):
        response = {
            "plan": [
                "sentiment_analysis",
                "text_classification",
                "summarize_text",
                "semantic_search",
                "filter_documents",
                "generate_embeddings",
                "research_models",
                "model_recommendation",
            ],
            "error": None,
            "final_answer": "Comprehensive test answer.",
            "latency": {"tool:sentiment_analysis": 120.0, "tool:model_recommendation": 1800.0},
            "tool_results": {
                "sentiment_analysis": {
                    "per_document": [{"id": "0", "label": "positive", "score": 0.99}],
                    "distribution": {"positive": 0.5, "negative": 0.5},
                    "skipped_count": 0,
                },
                "text_classification": {
                    "per_document": [{"id": "0", "label": "billing", "score": 0.8, "all_scores": {"billing": 0.8}}]
                },
                "summarize_text": {"summary": "A summary.", "source_document_ids": ["0", "1"], "chunked": False},
                "semantic_search": {"results": [{"id": "0", "text_excerpt": "excerpt", "score": 0.7}]},
                "filter_documents": {"document_ids": ["1"], "count": 1},
                "generate_embeddings": {"index_id": "x", "n_vectors": 5, "dim": 384, "built": True, "cached": False},
                "research_models": {
                    "evidence": [
                        {
                            "claim": "X is faster.",
                            "source_title": "HF card",
                            "source_url": "https://hf.co/x",
                            "snippet": "...",
                        }
                    ],
                    "found": True,
                },
                "model_recommendation": {
                    "recommendation": "distilbert-base-uncased-finetuned-sst-2-english",
                    "rationale": ["Small dataset favors pretrained."],
                    "measured_on_user_data": [
                        {
                            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                            "accuracy": 0.95,
                            "f1": 0.94,
                            "n_examples": 25,
                        }
                    ],
                    "measured_skip_reason": None,
                    "external_research": [
                        {
                            "claim": "X is faster.",
                            "source_title": "HF card",
                            "source_url": "https://hf.co/x",
                            "snippet": "...",
                        }
                    ],
                    "research_note": None,
                    "system_judgment": "Recommended based on dataset characteristics.",
                    "confidence_note": "High confidence.",
                    "fine_tune_note": "This system does not perform training or fine-tuning; this is guidance only.",
                    "degraded": False,
                },
            },
        }

        at = _app_with_history({"query": "Test query", "response": response})

        assert not at.exception

    def test_model_recommendation_shows_three_sections_and_disclaimer(self):
        response = {
            "plan": ["model_recommendation"],
            "error": None,
            "final_answer": "See recommendation.",
            "latency": {},
            "tool_results": {
                "model_recommendation": {
                    "recommendation": "distilbert-base-uncased-finetuned-sst-2-english",
                    "rationale": ["Reason one."],
                    "measured_on_user_data": [],
                    "measured_skip_reason": "No evaluation was run.",
                    "external_research": [],
                    "research_note": "External research was not requested for this query.",
                    "system_judgment": "Recommended based on dataset characteristics and stated constraints.",
                    "confidence_note": "Moderate confidence.",
                    "fine_tune_note": "This system does not perform training or fine-tuning; this is guidance only.",
                    "degraded": False,
                }
            },
        }

        at = _app_with_history({"query": "Should I use BERT or DistilBERT?", "response": response})

        assert not at.exception
        all_text = (
            " ".join(m.value for m in at.markdown)
            + " ".join(c.value for c in at.caption)
            + " ".join(i.value for i in at.info)
        )
        assert "does not perform training or fine-tuning" in all_text
        assert "No evaluation was run." in all_text
        assert "not requested" in all_text

    def test_degraded_model_recommendation_shows_warning(self):
        response = {
            "plan": ["model_recommendation"],
            "error": None,
            "final_answer": "Degraded answer.",
            "latency": {},
            "tool_results": {
                "model_recommendation": {
                    "recommendation": "distilbert-base-uncased-finetuned-sst-2-english",
                    "rationale": ["Rule-based reason."],
                    "measured_on_user_data": [],
                    "measured_skip_reason": "No evaluation was run.",
                    "external_research": [],
                    "research_note": "External research was not requested for this query.",
                    "system_judgment": "Recommended based on dataset characteristics: distilbert.",
                    "confidence_note": "Degraded.",
                    "fine_tune_note": "This system does not perform training or fine-tuning; this is guidance only.",
                    "degraded": True,
                }
            },
        }

        at = _app_with_history({"query": "Should I use BERT or DistilBERT?", "response": response})

        assert not at.exception
        assert any("unavailable" in w.value.lower() for w in at.warning)

    def test_research_not_found_shows_muted_state_not_error(self):
        response = {
            "plan": ["research_models"],
            "error": None,
            "final_answer": "No evidence available.",
            "latency": {},
            "tool_results": {"research_models": {"evidence": [], "found": False}},
        }

        at = _app_with_history({"query": "research this", "response": response})

        assert not at.exception
        assert any("no external evidence found" in i.value.lower() for i in at.info)
        assert not list(at.error)


class TestErrorRendering:
    def test_full_failure_shows_error_not_crash(self):
        response = {
            "plan": [],
            "error": "boom",
            "final_answer": "I ran into a problem while working on that: boom.",
            "latency": {},
            "tool_results": {},
        }

        at = _app_with_history({"query": "q", "response": response})

        assert not at.exception
        assert any("boom" in e.value for e in at.error)

    def test_partial_failure_shows_warning_and_partial_results(self):
        response = {
            "plan": ["sentiment_analysis", "filter_documents"],
            "error": "'filter_documents' failed: no results",
            "final_answer": "Partial answer.",
            "latency": {"tool:sentiment_analysis": 100.0},
            "tool_results": {
                "sentiment_analysis": {
                    "per_document": [],
                    "distribution": {"positive": 1.0, "negative": 0.0},
                    "skipped_count": 0,
                }
            },
        }

        at = _app_with_history({"query": "q", "response": response})

        assert not at.exception
        assert any("did not complete" in w.value.lower() for w in at.warning)
        assert any("Partial answer." in m.value for m in at.markdown)


class TestSidebar:
    def test_corpus_info_renders_in_sidebar(self):
        at = _app_with_history()

        assert not at.exception
        assert any(m.label == "Documents" and m.value == "5" for m in at.metric)
