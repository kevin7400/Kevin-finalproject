from finance_lstm.pipeline import run_pipeline


def test_run_pipeline_is_callable():
    """
    Smoke test: ensure the main orchestrator can be imported and is callable.

    We do NOT actually run the full pipeline here (it would download data,
    train models, etc.). This is just to validate the public API and packaging.
    """
    assert callable(run_pipeline)
