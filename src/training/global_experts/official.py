"""Load pinned GMTNet with a narrowly scoped Windows import compatibility shim."""

import multiprocessing.context

from ...baselines.gmtnet.runner import _load_official_modules


def load_official_modules(root):
    context = multiprocessing.context
    missing = not hasattr(context, "ForkContext")
    if missing:
        # The pinned graphs.py imports this name but never uses it. No fork is emulated.
        context.ForkContext = object
    try:
        return _load_official_modules(root)
    finally:
        if missing:
            del context.ForkContext
