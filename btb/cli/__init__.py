# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""btb's command line.

`app` builds the parser and runs commands. Each command has its own module with its parser and handler. `args` has
the argument types and the shared model options, `model` loads models, and `console` handles output. Importing this
package does not import torch."""

from .app import main

__all__ = ["main"]
