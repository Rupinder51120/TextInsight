"""Lazy-loaded, cached Hugging Face pipeline registry.

Per docs/LATENCY_AND_PERFORMANCE.md §4: "Model instances: loaded once at process startup or on first use,
cached in a module-level registry for the life of the process (never reloaded per request)." This
module-level cache is explicitly the documented design (unlike the old notebook's module-level `df` that
CLAUDE.md §4 forbids) — it holds read-only model instances, never mutable request data, and is never written
to by anything other than get_pipeline itself.

Default model IDs match docs/TOOLS_AND_MODELS.md's "Default Pretrained Model Choices" table exactly.
"""

import threading
from typing import Any

_lock = threading.Lock()
_pipelines: dict[tuple[str, str], Any] = {}

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "valhalla/distilbart-mnli-12-3"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"


def get_pipeline(task: str, model: str, **kwargs: Any) -> Any:
    """Return a cached HF pipeline for (task, model), constructing it at most once per process."""
    key = (task, model)
    if key not in _pipelines:
        with _lock:
            if key not in _pipelines:
                from transformers import pipeline

                _pipelines[key] = pipeline(task, model=model, **kwargs)
    return _pipelines[key]


def get_sentiment_pipeline() -> Any:
    return get_pipeline("sentiment-analysis", SENTIMENT_MODEL)


def get_zero_shot_pipeline() -> Any:
    return get_pipeline("zero-shot-classification", ZERO_SHOT_MODEL)


def get_summarization_pipeline() -> Any:
    return get_pipeline("summarization", SUMMARIZATION_MODEL)
