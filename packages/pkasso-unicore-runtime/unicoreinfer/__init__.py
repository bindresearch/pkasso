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

# backwards compatibility to support `from unicoreinfer.X import Y`
from unicoreinfer.distributed import utils as distributed_utils
from unicoreinfer.logging import meters, metrics, progress_bar  # noqa

sys.modules["unicoreinfer.distributed_utils"] = distributed_utils
sys.modules["unicoreinfer.meters"] = meters
sys.modules["unicoreinfer.metrics"] = metrics
sys.modules["unicoreinfer.progress_bar"] = progress_bar

import unicoreinfer.losses  # noqa
import unicoreinfer.distributed  # noqa
import unicoreinfer.models  # noqa
import unicoreinfer.modules  # noqa
import unicoreinfer.tasks  # noqa
