"""目录扫描器 — 自动发现模型和数据集，并校验格式"""
import json
from pathlib import Path
from typing import Optional
from .validator import (
    validate_model_dir, validate_dataset_file,
    DATASET_EXTENSIONS,
)


def scan_models(models_dir: str = "models") -> list:
    """
    扫描 models/ 目录，返回可用模型列表（含校验信息）。

    返回列表每个元素:
        {
            "name": str, "path": str, "architecture": str,
            "model_type": str, "size": str,
            "valid": bool, "missing": [str], "warnings": [str], "errors": [str],
        }
    """
    base = Path(models_dir)
    if not base.exists():
        return []

    models = []
    seen = set()

    # 先检查 models/ 本身是否就是模型文件夹
    if (base / "config.json").exists():
        v = validate_model_dir(str(base))
        if v["valid"] or v["files"]:
            models.append({
                "name": base.name,
                "path": str(base),
                "architecture": v["architecture"],
                "model_type": v["model_type"],
                "size": v["total_size"],
                "valid": v["valid"],
                "missing": v["missing"],
                "warnings": v["warnings"],
                "errors": v["errors"],
            })
            seen.add(str(base.resolve()))

    # 再扫描子文件夹
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name == ".gitkeep":
            continue
        if str(d.resolve()) in seen:
            continue

        v = validate_model_dir(str(d))
        if v["valid"] or v["files"]:
            models.append({
                "name": d.name,
                "path": str(d),
                "architecture": v["architecture"],
                "model_type": v["model_type"],
                "size": v["total_size"],
                "valid": v["valid"],
                "missing": v["missing"],
                "warnings": v["warnings"],
                "errors": v["errors"],
            })

    return models


def scan_datasets(datasets_dir: str = "datasets") -> list:
    """
    扫描 datasets/ 目录（含子文件夹），返回可用数据集列表（含校验信息）。

    返回列表每个元素:
        {
            "name": str, "path": str, "size": str, "format": str,
            "valid": bool, "sample_count": int, "fields": [str],
            "warnings": [str], "errors": [str],
        }
    """
    base = Path(datasets_dir)
    if not base.exists():
        return []

    datasets = []

    def _scan_dir(directory: Path, depth: int = 0):
        if depth > 3:
            return
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(".") or entry.name == "README.md":
                continue
            if entry.is_file() and entry.suffix.lower() in DATASET_EXTENSIONS:
                v = validate_dataset_file(str(entry))
                rel = entry.relative_to(base)
                datasets.append({
                    "name": str(rel).replace("\\", "/"),
                    "path": str(entry),
                    "size": v["size"],
                    "format": v["format"],
                    "valid": v["valid"],
                    "sample_count": v["sample_count"],
                    "fields": v["fields"],
                    "warnings": v["warnings"],
                    "errors": v["errors"],
                })
            elif entry.is_dir() and depth < 2:
                _scan_dir(entry, depth + 1)

    _scan_dir(base)
    return datasets


def scan_outputs(outputs_dir: str = "outputs") -> list:
    """扫描 outputs/ 目录，返回已训练的模型列表"""
    base = Path(outputs_dir)
    if not base.exists():
        return []

    outputs = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name == ".gitkeep":
            continue
        adapter = d / "adapter_config.json"
        if adapter.exists():
            outputs.append({
                "name": d.name,
                "path": str(d),
            })

    return outputs


def pick_model(models_dir: str = "models") -> Optional[str]:
    """交互式选择模型"""
    models = scan_models(models_dir)
    if not models:
        print(f"  models/ 目录下没有找到模型")
        print(f"  请将 HuggingFace 模型文件夹放入 models/ 目录")
        return None

    valid = [m for m in models if m["valid"]]
    if not valid:
        print(f"  发现了 {len(models)} 个文件夹，但格式都不正确:")
        for m in models:
            print(f"    FAIL {m['name']}: {', '.join(m['errors']) if m['errors'] else '未知错误'}")
        return None

    if len(valid) == 1:
        m = valid[0]
        print(f"  自动选择模型: {m['name']} ({m['architecture']}, {m['size']})")
        return m["path"]

    print(f"\n  发现 {len(valid)} 个可用模型:")
    for i, m in enumerate(valid):
        status = "OK" if m["valid"] else "FAIL"
        print(f"    [{i+1}] {status} {m['name']} ({m['architecture']}, {m['size']})")

    while True:
        try:
            choice = input(f"  选择模型 [1-{len(valid)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(valid):
                return valid[idx]["path"]
        except (ValueError, EOFError):
            return None


def pick_dataset(datasets_dir: str = "datasets") -> Optional[str]:
    """交互式选择数据集"""
    datasets = scan_datasets(datasets_dir)
    if not datasets:
        print(f"  datasets/ 目录下没有找到数据集")
        print(f"  请将训练数据文件放入 datasets/ 目录")
        print(f"  支持格式: .json, .jsonl, .csv, .txt")
        return None

    if len(datasets) == 1:
        d = datasets[0]
        status = "OK" if d["valid"] else "FAIL"
        print(f"  自动选择数据集: [{status}] {d['name']} ({d['size']}, {d['format']})")
        return d["path"]

    print(f"\n  发现 {len(datasets)} 个数据集:")
    for i, d in enumerate(datasets):
        status = "OK" if d["valid"] else "FAIL"
        print(f"    [{i+1}] {status} {d['name']} ({d['size']}, {d['format']}, {d['sample_count']}条)")

    while True:
        try:
            choice = input(f"  选择数据集 [1-{len(datasets)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]["path"]
        except (ValueError, EOFError):
            return None


def print_scan_report():
    """打印扫描报告（含校验信息）"""
    from .validator import print_validation_report
    print_validation_report()
