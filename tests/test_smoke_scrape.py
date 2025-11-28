import importlib


def test_pipeline_importable():
    mod = importlib.import_module('hidden_gems.scrape.pipeline')
    assert hasattr(mod, 'run_pipeline')
