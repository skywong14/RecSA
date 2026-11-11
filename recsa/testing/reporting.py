from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def collect_env() -> Dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception as exc:
        triton_version = f"unavailable: {exc}"

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception as exc:
            gpu_name = f"unavailable: {exc}"

    return {
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys.executable": sys.executable,
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


def dump_env(path: str | Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = collect_env()
    if extra:
        data.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n")
    return data


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n")


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    flat = diff.reshape(-1)
    worst_flat = int(torch.argmax(flat).item()) if flat.numel() else -1
    worst_index = list(torch.unravel_index(torch.tensor(worst_flat), diff.shape)) if flat.numel() else []
    return {
        "max_abs": float(flat.max().item()) if flat.numel() else 0.0,
        "mean_abs": float(flat.mean().item()) if flat.numel() else 0.0,
        "p99_abs": float(torch.quantile(flat, 0.99).item()) if flat.numel() else 0.0,
        "p999_abs": float(torch.quantile(flat, 0.999).item()) if flat.numel() else 0.0,
        "worst_index": [int(x.item()) for x in worst_index],
        "numel": int(flat.numel()),
    }


def ensure_report_files(report_dir: str | Path, config: Optional[Dict[str, Any]] = None) -> Path:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    env_path = root / "env.json"
    if not env_path.exists():
        dump_env(env_path)
    config_path = root / "config.json"
    if config is not None:
        write_json(config_path, config)
    elif not config_path.exists():
        write_json(config_path, {})
    (root / "results.jsonl").touch(exist_ok=True)
    (root / "stdout.log").touch(exist_ok=True)
    return root


def append_result(report_dir: str | Path, record: Dict[str, Any]) -> None:
    JsonlWriter(Path(report_dir) / "results.jsonl").write(record)


def command_record(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    args = list(argv) if argv is not None else sys.argv
    return {
        "argv": args,
        "command": " ".join(args),
    }
