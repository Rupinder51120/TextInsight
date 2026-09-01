"""Model registry caching — docs/FIVE_DAY_BUILD_PLAN.md Day 2: "model-registry caching test (second call to
a tool doesn't reload the model)". Patches transformers.pipeline itself so no real model download is needed
to verify the caching mechanics."""

from unittest.mock import MagicMock

from models.registry import _pipelines, get_pipeline


def test_second_call_with_same_task_and_model_does_not_reconstruct_pipeline(monkeypatch):
    _pipelines.clear()
    mock_pipeline_fn = MagicMock(return_value="fake-pipeline-instance")
    monkeypatch.setattr("transformers.pipeline", mock_pipeline_fn)

    first = get_pipeline("sentiment-analysis", "some/model")
    second = get_pipeline("sentiment-analysis", "some/model")

    assert first is second
    mock_pipeline_fn.assert_called_once()


def test_different_model_gets_its_own_cache_entry(monkeypatch):
    _pipelines.clear()
    mock_pipeline_fn = MagicMock(side_effect=lambda task, model, **kwargs: f"{task}:{model}")
    monkeypatch.setattr("transformers.pipeline", mock_pipeline_fn)

    a = get_pipeline("sentiment-analysis", "model-a")
    b = get_pipeline("sentiment-analysis", "model-b")

    assert a != b
    assert mock_pipeline_fn.call_count == 2
