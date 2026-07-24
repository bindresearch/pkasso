# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# Backwards-compatible aliases retained within the private runtime package.
from .distributed import utils as distributed_utils
from .logging import meters, metrics, progress_bar  # noqa

sys.modules[f"{__name__}.distributed_utils"] = distributed_utils
sys.modules[f"{__name__}.meters"] = meters
sys.modules[f"{__name__}.metrics"] = metrics
sys.modules[f"{__name__}.progress_bar"] = progress_bar

from . import distributed, losses, models, modules, tasks  # noqa: E402,F401
