import importlib
from hidden_gems.io import raw_path, processed_path, models_dir

p = importlib.import_module('hidden_gems.scrape.pipeline')
print('pipeline:', hasattr(p, 'run_pipeline'))

m = importlib.import_module('hidden_gems.ml.train')
print('ml train:', hasattr(m, 'main'))

b = importlib.import_module('hidden_gems.backend.app')
print('backend app:', hasattr(b, 'app'))

print('raw_dir exists:', raw_path('dummy').parent.exists())
print('processed_dir exists:', processed_path('merged_combined.csv').parent.exists())
print('models dir:', models_dir())
