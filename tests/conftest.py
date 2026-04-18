"""Force FAISS into single-thread mode before torch is imported.

On macOS, torch's bundled OpenMP symbols collide with FAISS's, causing
`IndexFlatIP.search` to hard-crash once both libraries share a process.
Importing FAISS first and pinning its thread count to 1 sidesteps the clash.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402,F401

faiss.omp_set_num_threads(1)
