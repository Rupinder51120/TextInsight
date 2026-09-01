"""Root pytest conftest.

KMP_DUPLICATE_LIB_OK works around a known macOS crash-on-interpreter-exit when both faiss-cpu and
PyTorch (loaded transitively by transformers/sentence-transformers) initialize their own bundled OpenMP
runtimes in the same process — harmless (it doesn't affect computed results, only process teardown), but
it can kill a test run. Must be set before those libraries are imported anywhere, so this has to live in
the root conftest.py (loaded first) rather than in application code.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
