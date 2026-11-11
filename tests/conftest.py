from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from recsa.testing.reporting import append_result, ensure_report_files


def pytest_configure(config: pytest.Config) -> None:
    report_dir = os.environ.get("RECSA_REPORT_DIR")
    if report_dir:
        ensure_report_files(report_dir, {"pytest_args": config.invocation_params.args})


@pytest.fixture
def record_m1_result(request):
    report_dir = os.environ.get("RECSA_REPORT_DIR")

    def _record(record: Dict[str, Any]) -> None:
        if not report_dir:
            return
        payload = {
            "test": request.node.nodeid,
            **record,
        }
        append_result(Path(report_dir), payload)

    return _record


@pytest.fixture
def record_m2_result(request):
    report_dir = os.environ.get("RECSA_REPORT_DIR")

    def _record(record: Dict[str, Any]) -> None:
        if not report_dir:
            return
        payload = {
            "test": request.node.nodeid,
            **record,
        }
        append_result(Path(report_dir), payload)

    return _record


@pytest.fixture
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RecSA Triton tests")
    return torch.device("cuda")


DEPLOYED_PREFILL_PINS = (
    ("recsa.parallel", "ACC_DOT_MODE", 0),
    ("recsa.parallel", "ACC_FP16", False),
    ("recsa.parallel", "MERGE_ACC", False),
    ("recsa.parallel", "UNION_LEAF", False),
    ("recsa.parallel", "UNION_MID", False),
    ("recsa.parallel", "FAST_SELECT", False),
    ("recsa.parallel", "FAST_TOP_SELECT", False),
    ("recsa.parallel", "SEL_IEEE", False),
)


def pin_deployed_prefill(mp: pytest.MonkeyPatch) -> None:
    import importlib

    for mod_name, attr, value in DEPLOYED_PREFILL_PINS:
        mp.setattr(importlib.import_module(mod_name), attr, value, raising=True)


@pytest.fixture
def deployed_prefill(monkeypatch):
    """Run the test on the deployed eval prefill (DEPLOYED_PREFILL_PINS) regardless of RECSA_PREFILL_FAST."""
    pin_deployed_prefill(monkeypatch)
    return monkeypatch
