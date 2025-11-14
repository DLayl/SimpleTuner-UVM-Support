#!/usr/bin/env python3
"""
Lightweight GH200 diagnostic script.

This utility is meant to be copied to the GH200 Jupyter environment and
executed there to confirm that the patched PyTorch build exposes the
expected Unified Virtual Memory behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import psutil
import torch


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def check_uvm_env() -> Dict[str, object]:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    enabled = "use_uvm:True" in conf
    ratio = None
    access_pattern = None
    if conf:
        for part in conf.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "uvm_oversubscription_ratio":
                try:
                    ratio = float(value)
                except ValueError:
                    ratio = value
            elif key == "uvm_access_pattern":
                access_pattern = value.lower()
    return {
        "enabled": enabled,
        "raw": conf,
        "oversubscription_ratio": ratio,
        "access_pattern": access_pattern,
    }


def run_subprocess(cmd: List[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
        return result.stdout.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def gather_system_info() -> Dict[str, object]:
    info: Dict[str, object] = {}

    info["architecture"] = run_subprocess(["uname", "-m"])
    info["kernel"] = run_subprocess(["uname", "-r"])
    info["cpu_count"] = psutil.cpu_count(logical=True)
    info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    info["available_ram_gb"] = round(psutil.virtual_memory().available / (1024**3), 1)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
        info["cuda_version"] = torch.version.cuda
        info["driver"] = run_subprocess(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    else:
        info["gpu_name"] = "<cuda not available>"

    return info


def test_uvm_allocation(scale: float = 1.2) -> Dict[str, object]:
    results: Dict[str, object] = {"requested_scale": scale}
    if not torch.cuda.is_available():
        results["status"] = "cuda_unavailable"
        return results

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    target_bytes = int(props.total_memory * scale)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    try:
        tensor = torch.empty(target_bytes, dtype=torch.uint8, device="cuda")
    except RuntimeError as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        return results

    alloc_time = time.time() - start

    results.update(
        {
            "status": "success",
            "allocated_gb": round(tensor.numel() / (1024**3), 2),
            "allocation_time_s": round(alloc_time, 3),
        }
    )

    fill_start = time.time()
    tensor.fill_(0x42)
    torch.cuda.synchronize()
    fill_time = time.time() - fill_start

    results["fill_bandwidth_gbps"] = round((tensor.numel() / (1024**3)) / fill_time, 2) if fill_time else None

    del tensor
    torch.cuda.empty_cache()
    return results


def compile_report(quota_scale: float, skip_allocation: bool) -> Dict[str, object]:
    report = {
        "uvm": check_uvm_env(),
        "system": gather_system_info(),
        "torch_version": torch.__version__,
    }
    if not skip_allocation:
        report["uvm_allocation_test"] = test_uvm_allocation(scale=quota_scale)
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GH200 readiness diagnostics.")
    parser.add_argument(
        "--oversubscription-scale",
        type=float,
        default=1.2,
        help="Allocation multiple relative to GPU memory to test UVM (default: 1.2x).",
    )
    parser.add_argument(
        "--skip-allocation",
        action="store_true",
        help="Skip the oversubscription allocation test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gh200_diagnostic_report.json"),
        help="Path to write the JSON report (default: gh200_diagnostic_report.json).",
    )
    args = parser.parse_args(argv)

    report = compile_report(args.oversubscription_scale, args.skip_allocation)

    print("\n=== GH200 DIAGNOSTIC SUMMARY ===")
    uvm = report["uvm"]
    print(f"UVM enabled: {uvm['enabled']} (config='{uvm['raw']}')")
    if uvm.get("oversubscription_ratio"):
        print(f"Requested oversubscription ratio: {uvm['oversubscription_ratio']}")
    if uvm.get("access_pattern"):
        print(f"Access pattern: {uvm['access_pattern']}")

    sys_info = report["system"]
    print(
        f"System: arch={sys_info['architecture']} cpu_cores={sys_info['cpu_count']} "
        f"ram={sys_info['total_ram_gb']}GB (available={sys_info['available_ram_gb']}GB)"
    )
    print(f"GPU: {sys_info['gpu_name']}, CUDA={sys_info.get('cuda_version', 'n/a')}, driver={sys_info.get('driver')}")

    allocation = report.get("uvm_allocation_test")
    if allocation:
        print(f"UVM allocation status: {allocation['status']}")
        if allocation.get("status") == "success":
            print(
                f"  Allocated {allocation['allocated_gb']} GB in {allocation['allocation_time_s']}s "
                f"(fill bandwidth ≈ {allocation['fill_bandwidth_gbps']} GB/s)"
            )
        else:
            print(f"  Error: {allocation.get('error')}")

    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
