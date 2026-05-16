"""目录扫描器 — 自动发现模型和数据集"""
import json
from pathlib import Path
from typing import Optional


def scan_models(models_dir: str = "models") -> list:
    """扫描 models/ 目录，返回可用模型列表"""
    base = Path(models_dir)
    if not base.exists():
        return []

    models = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        config = d / "config.json"
        if not config.exists():
            continue

        try:
            cfg = json.loads(config.read_text(encoding="utf-8"))
            name = d.name
            arch = cfg.get("architectures", ["Unknown"])[0] if "architectures" in cfg else "Unknown"
            params = cfg.get("num_hidden_layers", "?") if "num_hidden_layers" in cfg else "?"
            models.append({
                "name": name,
                "path": str(d),
                "architecture": arch,
            })
        except Exception:
            models.append({"name": d.name, "path": str(d), "architecture": "Unknown"})

    return models


def scan_datasets(datasets_dir: str = "datasets") -> list:
    """扫描 datasets/ 目录，返回可用数据集列表"""
    base = Path(datasets_dir)
    if not base.exists():
        return []

    datasets = []
    for f in sorted(base.iterdir()):
        if f.is_file() and f.suffix.lower() in (".json", ".jsonl", ".csv", ".txt"):
            if f.name in ("README.md", ".gitkeep"):
                continue
            size_kb = f.stat().st_size / 1024
            datasets.append({
                "name": f.name,
                "path": str(f),
                "size": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB",
                "format": f.suffix.lower().lstrip("."),
            })

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

    if len(models) == 1:
        m = models[0]
        print(f"  自动选择模型: {m['name']} ({m['architecture']})")
        return m["path"]

    print(f"\n  发现 {len(models)} 个模型:")
    for i, m in enumerate(models):
        print(f"    [{i+1}] {m['name']} ({m['architecture']})")

    while True:
        try:
            choice = input(f"  选择模型 [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["path"]
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
        print(f"  自动选择数据集: {d['name']} ({d['size']})")
        return d["path"]

    print(f"\n  发现 {len(datasets)} 个数据集:")
    for i, d in enumerate(datasets):
        print(f"    [{i+1}] {d['name']} ({d['size']})")

    while True:
        try:
            choice = input(f"  选择数据集 [1-{len(datasets)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]["path"]
        except (ValueError, EOFError):
            return None


def print_scan_report():
    """打印扫描报告"""
    print("\n" + "=" * 50)
    print("  项目目录扫描")
    print("=" * 50)

    models = scan_models()
    print(f"\n  models/ ({len(models)} 个模型)")
    for m in models:
        print(f"    - {m['name']} ({m['architecture']})")
    if not models:
        print(f"    (空 — 请放入 HuggingFace 模型文件夹)")

    datasets = scan_datasets()
    print(f"\n  datasets/ ({len(datasets)} 个数据集)")
    for d in datasets:
        print(f"    - {d['name']} ({d['size']}, {d['format']})")
    if not datasets:
        print(f"    (空 — 请放入 .json / .jsonl / .csv / .txt 文件)")

    outputs = scan_outputs()
    print(f"\n  outputs/ ({len(outputs)} 个已训练模型)")
    for o in outputs:
        print(f"    - {o['name']}")
    if not outputs:
        print(f"    (空 — 训练完成后会保存在这里)")

    print()
