"""
格式校验器 — 自动检测 models/ 和 datasets/ 下的文件格式是否正确
============================================================
模型校验: 检查 HuggingFace 标准模型文件夹的必要文件
数据集校验: 检查 JSON/JSONL/CSV/TXT 文件格式和内容完整性
"""

import json
import csv
from pathlib import Path
from typing import Optional, Tuple


# ── 模型校验 ─────────────────────────────────────────────

REQUIRED_MODEL_FILES = ["config.json"]
WEIGHT_GLOBS = ["*.safetensors", "*.bin", "*.pt", "*.h5", "*.ckpt"]
TOKENIZER_FILES = [
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",
]
OPTIONAL_MODEL_FILES = ["generation_config.json", "preprocessor_config.json"]
CONFIG_KEYS = ["architectures", "model_type"]


def validate_model_dir(model_path: str) -> dict:
    """
    校验模型文件夹格式。

    返回:
        {
            "name": str,          # 模型名称
            "path": str,          # 路径
            "valid": bool,        # 是否可用
            "architecture": str,  # 模型架构
            "model_type": str,    # 模型类型
            "required_ok": bool,  # 必要文件是否齐全
            "total_size": str,    # 模型总大小
            "files": [str],       # 发现的模型文件
            "missing": [str],     # 缺失的必要文件
            "warnings": [str],    # 警告信息
            "errors": [str],      # 错误信息
        }
    """
    path = Path(model_path)
    result = {
        "name": path.name,
        "path": str(path),
        "valid": False,
        "architecture": "Unknown",
        "model_type": "Unknown",
        "required_ok": False,
        "total_size": "0 KB",
        "files": [],
        "missing": [],
        "warnings": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append("路径不存在")
        return result

    if not path.is_dir():
        result["errors"].append("不是文件夹")
        return result

    # 收集所有文件名
    all_files = {f.name: f for f in path.iterdir() if f.is_file()}
    result["files"] = sorted(all_files.keys())

    # 计算总大小
    total_bytes = sum(f.stat().st_size for f in all_files.values())
    if total_bytes < 1024:
        result["total_size"] = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        result["total_size"] = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        result["total_size"] = f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        result["total_size"] = f"{total_bytes / (1024 * 1024 * 1024):.1f} GB"

    # 1. 检查 config.json
    config_file = path / "config.json"
    if not config_file.exists():
        result["missing"].append("config.json")
        result["errors"].append("缺少 config.json（模型架构配置）")
    else:
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            if "architectures" in cfg:
                result["architecture"] = cfg["architectures"][0] if isinstance(cfg["architectures"], list) else str(cfg["architectures"])
            elif "model_type" in cfg:
                result["architecture"] = cfg["model_type"]
            result["model_type"] = cfg.get("model_type", "Unknown")
        except json.JSONDecodeError:
            result["errors"].append("config.json 不是有效 JSON")
        except Exception as e:
            result["errors"].append(f"读取 config.json 失败: {e}")

    # 2. 检查权重文件
    weight_files = []
    for glob_pat in WEIGHT_GLOBS:
        weight_files.extend(path.glob(glob_pat))
    # 过滤 index 文件
    weight_files = [f for f in weight_files if "index" not in f.name.lower()]
    if not weight_files:
        result["missing"].append("权重文件 (.safetensors / .bin / .pt)")
        result["errors"].append("未找到模型权重文件")
    else:
        weight_names = [f.name for f in weight_files]
        sharded = any(".safetensors" in w for w in weight_names) and len(weight_names) > 1
        if sharded:
            # 检查分片文件是否有 index
            index_exists = any("index" in f.name.lower() for f in path.glob("*.safetensors*"))
            if not index_exists:
                result["warnings"].append("分片权重缺少 index 文件")

    # 3. 检查 tokenizer
    tokenizer_found = [f for f in TOKENIZER_FILES if (path / f).exists()]
    if not tokenizer_found:
        result["missing"].append("tokenizer 文件")
        result["errors"].append("未找到 tokenizer 文件，无法做文本编码")
    elif "tokenizer_config.json" not in tokenizer_found:
        result["warnings"].append("缺少 tokenizer_config.json")

    # 4. 检查可选文件
    for opt_f in OPTIONAL_MODEL_FILES:
        if (path / opt_f).exists():
            pass  # 可选文件存在是好事

    # 5. 综合判断
    result["required_ok"] = len(result["missing"]) == 0
    result["valid"] = config_file.exists() and len(weight_files) > 0 and len(tokenizer_found) > 0

    return result


# ── 数据集校验 ───────────────────────────────────────────

DATASET_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}
SHAREGPT_KEYS = {"messages"}
ALPACA_KEYS = {"instruction", "output"}
ALPACA_ALT_KEYS = {"instruction", "response"}
JSONL_KEYS = {"question", "answer"}
JSONL_ALT_KEYS = {"prompt", "completion"}
JSONL_ALT2_KEYS = {"input", "output"}


def _check_json_format(item: dict) -> Optional[str]:
    """判断单条 JSON 数据的格式类型"""
    keys = set(item.keys())
    if SHAREGPT_KEYS.issubset(keys):
        msgs = item.get("messages", [])
        if isinstance(msgs, list) and len(msgs) > 0:
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
                return "sharegpt"
    if ALPACA_KEYS.issubset(keys) or ALPACA_ALT_KEYS.issubset(keys):
        return "alpaca"
    if JSONL_KEYS.issubset(keys) or JSONL_ALT_KEYS.issubset(keys) or JSONL_ALT2_KEYS.issubset(keys):
        return "jsonl"
    return None


def validate_dataset_file(dataset_path: str) -> dict:
    """
    校验数据集文件格式和内容。

    返回:
        {
            "name": str,
            "path": str,
            "valid": bool,
            "format": str,         # 检测到的格式
            "sample_count": int,   # 样本数
            "size": str,
            "fields": [str],       # 检测到的字段
            "errors": [str],
            "warnings": [str],
            "first_sample": dict,  # 第一条样本预览
        }
    """
    path = Path(dataset_path)
    result = {
        "name": path.name,
        "path": str(path),
        "valid": False,
        "format": "unknown",
        "sample_count": 0,
        "size": "0 KB",
        "fields": [],
        "errors": [],
        "warnings": [],
        "first_sample": None,
    }

    if not path.exists():
        result["errors"].append("文件不存在")
        return result
    if not path.is_file():
        result["errors"].append("不是文件")
        return result

    size_bytes = path.stat().st_size
    if size_bytes < 1024:
        result["size"] = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        result["size"] = f"{size_bytes / 1024:.1f} KB"
    else:
        result["size"] = f"{size_bytes / (1024 * 1024):.1f} MB"

    suffix = path.suffix.lower()
    result["format"] = suffix.lstrip(".")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        result["errors"].append("文件编码不是 UTF-8")
        return result
    except Exception as e:
        result["errors"].append(f"读取文件失败: {e}")
        return result

    if not content.strip():
        result["errors"].append("文件为空")
        return result

    # ── JSON ──
    if suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            result["errors"].append(f"JSON 解析失败: {e}")
            return result

        if isinstance(data, list):
            result["sample_count"] = len(data)
            if len(data) == 0:
                result["errors"].append("JSON 数组为空")
                return result
            first = data[0]
            if isinstance(first, dict):
                detected = _check_json_format(first)
                if detected:
                    result["format"] = detected
                    result["fields"] = list(first.keys())
                    result["first_sample"] = first
                    result["valid"] = True
                else:
                    result["warnings"].append(f"未识别的 JSON 结构，字段: {list(first.keys())}")
                    result["fields"] = list(first.keys())
            else:
                result["warnings"].append(f"JSON 数组元素不是对象，而是 {type(first).__name__}")
        elif isinstance(data, dict):
            result["warnings"].append("JSON 是对象而非数组，可能不是标准数据集格式")
        else:
            result["errors"].append("JSON 格式不符合预期")

    # ── JSONL ──
    elif suffix == ".jsonl":
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        result["sample_count"] = len(lines)
        if len(lines) == 0:
            result["errors"].append("JSONL 文件为空")
            return result

        parse_errors = 0
        for i, line in enumerate(lines):
            try:
                item = json.loads(line)
                if i == 0:
                    if isinstance(item, dict):
                        detected = _check_json_format(item)
                        if detected:
                            result["format"] = detected
                        result["fields"] = list(item.keys())
                        result["first_sample"] = item
            except json.JSONDecodeError:
                parse_errors += 1

        if parse_errors > 0:
            result["errors"].append(f"{parse_errors}/{len(lines)} 行不是有效 JSON")
        else:
            result["valid"] = True
            if result["fields"] and not _check_json_format(json.loads(lines[0])):
                result["warnings"].append(f"JSONL 字段无法匹配已知格式: {result['fields']}")

    # ── CSV ──
    elif suffix == ".csv":
        try:
            reader = csv.reader(content.splitlines())
            rows = list(reader)
        except Exception as e:
            result["errors"].append(f"CSV 解析失败: {e}")
            return result

        if len(rows) < 2:
            result["errors"].append("CSV 至少需要表头 + 一行数据")
            return result

        header = rows[0]
        result["fields"] = header
        result["sample_count"] = len(rows) - 1

        # 检查常见列名
        known = {"question", "answer", "prompt", "completion", "instruction", "input", "output", "response"}
        matched = set(h.lower() for h in header) & known
        if not matched:
            result["warnings"].append(f"CSV 列名未匹配已知字段，将用前两列训练: {header}")
        else:
            result["valid"] = True

        result["first_sample"] = dict(zip(header, rows[1] if len(rows) > 1 else []))

    # ── TXT ──
    elif suffix == ".txt":
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        result["sample_count"] = len(paragraphs) if paragraphs else 1
        if len(paragraphs) <= 1:
            result["warnings"].append("文本未分段落(双换行)，整个文件将作为一个样本")
        result["valid"] = True
        preview = content[:200].replace("\n", "\\n")
        result["first_sample"] = {"text": preview + ("..." if len(content) > 200 else "")}

    return result


# ── 批量校验 ─────────────────────────────────────────────

def validate_all_models(models_dir: str = "models", max_depth: int = 3) -> list:
    """递归扫描并校验 models/ 下所有模型"""
    base = Path(models_dir)
    if not base.exists():
        return []

    results = []
    seen = set()

    # 先检查 models/ 本身是否就是模型文件夹
    if (base / "config.json").exists():
        result = validate_model_dir(str(base))
        if result["valid"] or result["files"]:
            results.append(result)
            seen.add(str(base.resolve()))

    def _scan_dir(directory: Path, depth: int = 0):
        if depth > max_depth:
            return
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith(".") or entry.name in (".gitkeep", "README.md"):
                continue
            resolved = str(entry.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)

            result = validate_model_dir(str(entry))
            if result["valid"] or result["files"]:
                try:
                    rel = entry.relative_to(base)
                    result["name"] = str(rel).replace("\\", "/")
                except ValueError:
                    pass
                results.append(result)
            _scan_dir(entry, depth + 1)

    _scan_dir(base)
    return results


def validate_all_datasets(datasets_dir: str = "datasets") -> list:
    """扫描并校验 datasets/ 下所有数据集"""
    base = Path(datasets_dir)
    if not base.exists():
        return []

    results = []

    def _scan_dir(directory: Path, depth: int = 0):
        if depth > 3:
            return
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(".") or entry.name == "README.md":
                continue
            if entry.is_file() and entry.suffix.lower() in DATASET_EXTENSIONS:
                result = validate_dataset_file(str(entry))
                # 用相对路径作为名称
                rel = entry.relative_to(base)
                result["name"] = str(rel).replace("\\", "/")
                results.append(result)
            elif entry.is_dir() and depth < 2:
                _scan_dir(entry, depth + 1)

    _scan_dir(base)
    return results


def print_validation_report(models_dir: str = "models", datasets_dir: str = "datasets"):
    """打印完整的校验报告"""
    print("\n" + "=" * 60)
    print("  格式校验报告")
    print("=" * 60)

    # ── 模型 ──
    models = validate_all_models(models_dir)
    print(f"\n  [models/] ({len(models)} 个文件夹)")
    if not models:
        print("    (空)")
    for m in models:
        status_icon = "OK" if m["valid"] else "FAIL"
        name = m["name"]
        arch = m["architecture"]
        size = m["total_size"]
        print(f"    [{status_icon}] {name}")
        print(f"        架构: {arch}  大小: {size}")
        if m["errors"]:
            for err in m["errors"]:
                print(f"        [ERR] {err}")
        if m["warnings"]:
            for w in m["warnings"]:
                print(f"        [WARN] {w}")
        if m["valid"]:
            print(f"        文件: {', '.join(m['files'][:8])}{'...' if len(m['files']) > 8 else ''}")

    # ── 数据集 ──
    datasets = validate_all_datasets(datasets_dir)
    print(f"\n  [datasets/] ({len(datasets)} 个文件)")
    if not datasets:
        print("    (空 — 支持 .json / .jsonl / .csv / .txt)")
    for d in datasets:
        status_icon = "OK" if d["valid"] else "FAIL"
        name = d["name"]
        fmt = d["format"]
        cnt = d["sample_count"]
        size = d["size"]
        print(f"    [{status_icon}] {name}")
        print(f"        格式: {fmt}  样本数: {cnt}  大小: {size}")
        if d["fields"]:
            print(f"        字段: {d['fields']}")
        if d["errors"]:
            for err in d["errors"]:
                print(f"        [ERR] {err}")
        if d["warnings"]:
            for w in d["warnings"]:
                print(f"        [WARN] {w}")

    # ── 汇总 ──
    valid_models = sum(1 for m in models if m["valid"])
    valid_datasets = sum(1 for d in datasets if d["valid"])
    print(f"\n  {'─' * 40}")
    print(f"  模型: {valid_models}/{len(models)} 可用")
    print(f"  数据集: {valid_datasets}/{len(datasets)} 可用")
    print()
