"""
通用数据集加载器 — 支持所有主流格式
======================================
ShareGPT  → {"messages": [{"role":"user","content":"..."}, ...]}
Alpaca    → {"instruction": "...", "input": "...", "output": "..."}
OpenAI    → 同 ShareGPT
JSONL     → 每行一个 JSON 对象
CSV       → question/answer 或 instruction/input/output 列
Text      → 纯文本
Auto      → 自动检测格式

自动检测 → 统一转为 HuggingFace Dataset (含 messages 字段)
"""

import json, csv, io
from pathlib import Path
from typing import Optional, List
from datasets import Dataset, concatenate_datasets


def detect_format(file_path: str) -> str:
    """自动检测数据集格式"""
    path = Path(file_path)

    if path.is_dir():
        # 扫描目录内文件
        files = list(path.glob("*"))
        json_files = [f for f in files if f.suffix in (".json", ".jsonl")]
        if json_files:
            path = json_files[0]
        else:
            csv_files = [f for f in files if f.suffix == ".csv"]
            if csv_files:
                path = csv_files[0]
            else:
                txt_files = [f for f in files if f.suffix == ".txt"]
                if txt_files:
                    return "text"

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "text"

    # 读第一行判断 JSON 格式
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()

    if first.startswith("["):
        data = json.loads(open(path, "r", encoding="utf-8").read())
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if "messages" in item:
                return "sharegpt"
            if "instruction" in item and ("output" in item or "response" in item):
                return "alpaca"
        return "sharegpt"  # 默认当 ShareGPT

    # JSONL: 每行一个 JSON
    try:
        item = json.loads(first)
        if "messages" in item:
            return "sharegpt"
        if "instruction" in item:
            return "alpaca"
        if "question" in item or "prompt" in item:
            return "jsonl"
        return "jsonl"
    except json.JSONDecodeError:
        return "text"


def _load_sharegpt(path: str) -> Dataset:
    """ShareGPT/OpenAI: {"messages": [...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list([{"messages": item["messages"]} for item in data])


def _load_alpaca(path: str) -> Dataset:
    """Alpaca: {"instruction":"...", "input":"...", "output":"..."} → ShareGPT"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages_list = []
    for item in data:
        user_content = item.get("instruction", "")
        if item.get("input"):
            user_content += "\n\n" + item["input"]
        assistant_content = item.get("output") or item.get("response") or ""
        messages_list.append({
            "messages": [
                {"role": "user", "content": user_content.strip()},
                {"role": "assistant", "content": assistant_content.strip()},
            ]
        })
    return Dataset.from_list(messages_list)


def _load_jsonl(path: str) -> Dataset:
    """JSONL 格式: 每行一个 JSON (question/answer 或 prompt/completion)"""
    messages_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 尝试各种字段名
            q = item.get("question") or item.get("prompt") or item.get("input") or item.get("instruction") or ""
            a = item.get("answer") or item.get("response") or item.get("output") or item.get("completion") or ""
            if isinstance(a, dict):
                a = a.get("text", str(a))
            messages_list.append({
                "messages": [
                    {"role": "user", "content": str(q).strip()},
                    {"role": "assistant", "content": str(a).strip()},
                ]
            })
    return Dataset.from_list(messages_list)


def _load_csv(path: str) -> Dataset:
    """CSV: 自动匹配 question/answer 或 instruction/input/output 列"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 为空")

    headers = list(rows[0].keys())
    messages_list = []

    # 检测列模式
    has_q = any(h in headers for h in ("question", "prompt", "input", "instruction"))
    has_a = any(h in headers for h in ("answer", "response", "output", "completion"))
    has_instruction = "instruction" in headers

    for row in rows:
        if has_instruction:
            user_c = row.get("instruction", "")
            if row.get("input"):
                user_c += "\n\n" + row["input"]
            assistant_c = row.get("output") or row.get("response") or row.get("answer") or ""
        elif has_q and has_a:
            q_key = next((h for h in headers if h in ("question", "prompt", "input")), headers[0])
            a_key = next((h for h in headers if h in ("answer", "response", "output", "completion")), headers[-1])
            user_c = row.get(q_key, "")
            assistant_c = row.get(a_key, "")
        else:
            # 当纯文本: 第一列是输入，第二列是输出
            user_c = row.get(headers[0], "")
            assistant_c = row.get(headers[1], "") if len(headers) > 1 else ""

        messages_list.append({
            "messages": [
                {"role": "user", "content": str(user_c).strip()},
                {"role": "assistant", "content": str(assistant_c).strip()},
            ]
        })
    return Dataset.from_list(messages_list)


def _load_text(path: str) -> Dataset:
    """纯文本: 支持目录下多个 txt 文件，或单个 txt"""
    p = Path(path)
    texts = []
    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            with open(f, "r", encoding="utf-8") as fp:
                texts.append(fp.read().strip())
    else:
        # 尝试按双换行分割
        content = p.read_text(encoding="utf-8")
        parts = [t.strip() for t in content.split("\n\n") if t.strip()]
        if len(parts) == 1:
            texts = [content.strip()]
        else:
            texts = parts

    messages_list = [{"messages": [{"role": "assistant", "content": t}]} for t in texts if t]
    return Dataset.from_list(messages_list)


FORMAT_LOADERS = {
    "sharegpt": _load_sharegpt,
    "alpaca": _load_alpaca,
    "jsonl": _load_jsonl,
    "csv": _load_csv,
    "text": _load_text,
}


def load_dataset(
    file_path: str,
    format: str = "auto",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    通用数据集加载入口。

    参数:
        file_path: 数据集文件或目录路径
        format: "auto" 自动检测, 或指定 "sharegpt" / "alpaca" / "jsonl" / "csv" / "text"
        max_samples: 最大样本数
        shuffle: 是否打乱
        seed: 随机种子

    返回:
        HuggingFace Dataset，每个样本含 "messages" 字段
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"数据集不存在: {file_path}")

    # 自动检测格式
    if format == "auto":
        format = detect_format(file_path)
        print(f"  检测到格式: {format}")

    # 如果是目录且有多个同格式文件，全部加载
    if path.is_dir() and format != "text":
        files = sorted(path.glob(f"*.{format}")) if format in ("jsonl", "csv") else sorted(path.glob("*.json"))
        if len(files) > 1:
            parts = []
            for f in files:
                loader = FORMAT_LOADERS.get(format, _load_sharegpt)
                parts.append(loader(str(f)))
            ds = concatenate_datasets(parts)
        else:
            loader = FORMAT_LOADERS.get(format, _load_sharegpt)
            ds = loader(str(files[0]) if files else file_path)
    else:
        loader = FORMAT_LOADERS.get(format, _load_sharegpt)
        ds = loader(file_path)

    if shuffle and len(ds) > 0:
        ds = ds.shuffle(seed=seed)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"  加载完成: {len(ds)} 条样本")
    return ds


def list_supported_formats():
    """列出支持的格式"""
    return {
        "sharegpt": '{"messages": [{"role": "user/assistant", "content": "..."}]}',
        "alpaca": '{"instruction": "...", "input": "...", "output": "..."}',
        "jsonl": '每行 JSON: {"question": "...", "answer": "..."}',
        "csv": "CSV 表格: question,answer 或 instruction,input,output 列",
        "text": "纯文本文件",
        "auto": "自动检测",
    }
