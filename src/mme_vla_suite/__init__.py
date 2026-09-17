"""MME-VLA package defaults."""

import os

# These must be set before JAX initializes its GPU backend. Callers can still
# override either setting explicitly before importing this package.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
