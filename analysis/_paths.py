# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file _paths.py
# @brief Resolves run-dump paths for the analysis scripts.
#
import os

# Run dumps ship in the benchmark repository under baselines/. LODESTAR_DATA (or
# DATA) points at them; the default assumes the two repos are cloned side by side.
_DATA_ROOT = os.environ.get("LODESTAR_DATA") or os.environ.get("DATA") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "apollo-anomaly-atlas", "baselines")


def _d(*parts):
    """@brief Resolve a run-dump path under the configured data root.
    @param parts Path components, e.g. _d("full_log.jsonl").
    @return Absolute path inside LODESTAR_DATA / DATA / the default baselines dir.
    """
    return os.path.join(_DATA_ROOT, *parts)
