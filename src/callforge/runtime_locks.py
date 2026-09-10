"""Process-local guards for memory-heavy model runtimes."""
from __future__ import annotations

import threading


# CallForge's UI may process more than one file concurrently, but the configured
# Apple Silicon target has only enough unified memory for one heavy model stage.
HEAVY_MODEL_LOCK = threading.Semaphore(1)
