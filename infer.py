#!/usr/bin/env python
"""
Universal Trainer — 推理入口
=============================

单推理模式 (Standalone Inference): 不依赖训练管线，直接加载模型进行推理。
支持 base model 纯推理，或加载 LoRA adapter 推理。

用法:
    # 自动扫描 models/ + outputs/，交互选择
    python infer.py

    # 指定模型和 LoRA
    python infer.py --model outputs/trained_model
    python infer.py --model outputs/trained_model --base models/Qwen2-0.5B

    # 纯 base model 推理（不加载 LoRA）
    python infer.py --base-only --base models/Qwen2-0.5B

    # 单次问答
    python infer.py --model outputs/my_lora -q "What is AI?"

    # 测试模式
    python infer.py --model outputs/my_lora --test

    # 流式输出
    python infer.py --model outputs/my_lora --stream

    # 多轮对话模式
    python infer.py --model outputs/my_lora --chat

    # 批量推理（从文件）
    python infer.py --model outputs/my_lora --batch questions.jsonl

    # 调整生成参数
    python infer.py --model outputs/my_lora --temperature 0.3 --max-tokens 256

    # CPU 模式
    python infer.py --base models/Qwen2-0.5B --no-4bit --backend cpu
"""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from core.infer_engine import (
    InferenceEngine,
    InferConfig,
    resolve_base_from_adapter,
    scan_for_models,
    scan_for_adapters,
    IS_ROCM,
)

# ── 颜色支持 ────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    C_RESET = Style.RESET_ALL
    C_RED = Fore.RED
    C_GREEN = Fore.GREEN
    C_YELLOW = Fore.YELLOW
    C_CYAN = Fore.CYAN
    C_BLUE = Fore.BLUE
    C_MAGENTA = Fore.MAGENTA
    C_DIM = Style.DIM
    C_BRIGHT = Style.BRIGHT
except ImportError:
    C_RESET = C_RED = C_GREEN = C_YELLOW = C_CYAN = C_BLUE = C_MAGENTA = C_DIM = C_BRIGHT = ""


def _info(msg: str):
    print(f"{msg}")


def _warn(msg: str):
    print(f"{C_YELLOW}⚠ {msg}{C_RESET}")


def _error(msg: str):
    print(f"{C_RED}✗ {msg}{C_RESET}", file=sys.stderr)


def _ok(msg: str):
    print(f"{C_GREEN}✓ {msg}{C_RESET}")


def _banner():
    """打印 banner"""
    backend_info = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            if IS_ROCM:
                backend_info = f"ROCm (HIP {torch.version.hip})"
            else:
                backend_info = f"CUDA"
            try:
                gpu = torch.cuda.get_device_name(0)
                backend_info += f" — {gpu}"
            except Exception:
                pass
        else:
            backend_info = "CPU"
    except Exception:
        pass
    print(f"\n{C_BRIGHT}{C_CYAN}╔════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BRIGHT}{C_CYAN}║{C_RESET}  {C_BRIGHT}Universal Trainer — 推理模式{C_RESET}          {C_BRIGHT}{C_CYAN}║{C_RESET}")
    print(f"{C_BRIGHT}{C_CYAN}║{C_RESET}  {C_DIM}backend: {backend_info}{' ' * (30 - len(backend_info))}{C_RESET}  {C_BRIGHT}{C_CYAN}║{C_RESET}")
    print(f"{C_BRIGHT}{C_CYAN}╚════════════════════════════════════════════╝{C_RESET}\n")


# ── 交互式选择 ──────────────────────────────────────────────

def _interactive_select(items: list, label: str, fmt_fn=None) -> Optional[dict]:
    """通用交互式选择：单项自动确认，多项展示编号列表。"""
    if not items:
        return None
    if len(items) == 1:
        print(f"{C_GREEN}自动选择{label}:{C_RESET} {fmt_fn(items[0]) if fmt_fn else items[0].get('name', items[0])}")
        return items[0]
    print(f"\n{C_CYAN}发现 {len(items)} 个{label}:{C_RESET}")
    for i, item in enumerate(items):
        desc = fmt_fn(item) if fmt_fn else str(item.get("name", item))
        print(f"  {C_BRIGHT}[{i+1}]{C_RESET} {desc}")
    while True:
        try:
            choice = input(f"  选择{label} [1-{len(items)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"{C_RED}  输入 1-{len(items)} 之间的数字{C_RESET}")
        except (ValueError, IndexError, EOFError):
            print(f"{C_RED}  输入无效，请重试{C_RESET}")


# ── 主程序 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python infer.py",
        description="Universal Trainer — 独立推理模式 (Standalone Inference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python infer.py                                          # 交互式选择
  python infer.py --model outputs/my_lora                   # 指定LoRA
   python infer.py --model outputs/my_lora --no-stream       # 禁用流式
   python infer.py --model outputs/my_lora -q "What is AI?"  # 单次问答（默认流式）
  python infer.py --model outputs/my_lora --chat            # 多轮对话
  python infer.py --model outputs/my_lora --batch qs.jsonl  # 批量推理
  python infer.py --base-only --base models/Qwen2-0.5B      # 纯base推理
  python infer.py --base models/Qwen2-0.5B --no-4bit --backend cpu  # CPU模式
        """,
    )

    # ── 模型参数 ──
    parser.add_argument("--model", type=str, default=None,
                        help="LoRA adapter 路径 (outputs/xxx)")
    parser.add_argument("--base", type=str, default=None,
                        help="基座模型路径 (models/xxx 或 HF ID)")
    parser.add_argument("--base-only", action="store_true",
                        help="纯 base model 推理，不加载 LoRA")
    parser.add_argument("--no-4bit", action="store_true",
                        help="禁用 4-bit 量化 (CPU 模式自动禁用)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "cpu"],
                        help="计算后端 (默认: auto)")

    # ── 运行模式 ──
    parser.add_argument("-q", "--question", type=str, default=None,
                        help="单次问答")
    parser.add_argument("--test", action="store_true",
                        help="运行内置测试题")
    parser.add_argument("--no-stream", action="store_true",
                        help="禁用流式逐字输出（默认开启）")
    parser.add_argument("--chat", action="store_true",
                        help="多轮对话模式")
    parser.add_argument("--batch", type=str, default=None, metavar="FILE",
                        help="批量推理: 从 JSONL/JSON 文件读取问题列表")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="设置系统提示词")

    # ── 生成参数 ──
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成 token 数 (默认: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度 (默认: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="nucleus sampling (默认: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="top-k sampling (默认: 50)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="重复惩罚 (默认: 1.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--no-sample", action="store_true",
                        help="禁用采样 (do_sample=False, 贪心解码)")

    # ── 其他 ──
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，减少日志输出")
    parser.add_argument("--scan", action="store_true",
                        help="扫描并展示可用的模型和 LoRA adapter")

    args = parser.parse_args()

    # ── 扫描模式 ──
    if args.scan:
        _banner()
        print(f"{C_CYAN}◆ 基座模型 (models/){C_RESET}")
        models = scan_for_models()
        if not models:
            print(f"  {C_DIM}(未找到基座模型，请放入 models/ 目录){C_RESET}")
        else:
            for m in models:
                status = f"{C_GREEN}OK{C_RESET}" if m["valid"] else f"{C_RED}FAIL{C_RESET}"
                print(f"  [{status}] {m['name']} ({m.get('architecture', '?')}, {m.get('size', '?')})")

        print(f"\n{C_CYAN}◆ LoRA Adapter (outputs/){C_RESET}")
        adapters = scan_for_adapters()
        if not adapters:
            print(f"  {C_DIM}(未找到 LoRA adapter，请先训练模型){C_RESET}")
        else:
            for a in adapters:
                print(f"  {C_GREEN}●{C_RESET} {a['name']} (rank={a.get('rank', '?')}, base={a.get('base_model', '?')})")
        return

    # ── 解析模型路径 ──
    base_path: Optional[str] = args.base
    adapter_path: Optional[str] = args.model if not args.base_only else None

    # 扫描并交互选择
    if adapter_path is None and not args.base_only:
        adapters = scan_for_adapters()
        if adapters:
            chosen = _interactive_select(
                adapters, "LoRA adapter",
                fmt_fn=lambda x: f"{x['name']} (rank={x.get('rank', '?')}, base={x.get('base_model', '?')[:50]})",
            )
            if chosen:
                adapter_path = str(Path(chosen["path"]).resolve())
        else:
            if args.base is None:
                _warn("未找到 LoRA adapter，将尝试纯 base model 推理")

    # 校验 adapter 路径
    if adapter_path:
        p = Path(adapter_path)
        if not p.exists():
            _error(f"LoRA 路径不存在: {adapter_path}")
            adapters = scan_for_adapters()
            if adapters:
                for a in adapters:
                    print(f"    可用: {a['path']}")
            sys.exit(1)
        if not (p / "adapter_config.json").exists():
            _error(f"{adapter_path} 不是有效的 LoRA adapter（缺少 adapter_config.json）")
            sys.exit(1)

    # 解析基座模型
    if base_path is None:
        # 从 adapter 配置自动检测
        if adapter_path:
            detected = resolve_base_from_adapter(adapter_path)
            if detected:
                bp = Path(detected)
                if bp.exists():
                    base_path = str(bp.resolve())
                    _ok(f"从 adapter 配置检测到基座模型: {base_path}")
                else:
                    base_path = detected
                    _ok(f"从 adapter 配置检测到基座模型: {detected} (在线)")
            else:
                models = scan_for_models()
                if models:
                    valid = [m for m in models if m["valid"]]
                    if valid:
                        chosen = _interactive_select(
                            valid, "基座模型",
                            fmt_fn=lambda m: f"{m['name']} ({m.get('architecture', '?')}, {m.get('size', '?')})",
                        )
                        if chosen:
                            base_path = str(Path(chosen["path"]).resolve())
                if base_path is None:
                    _error("无法确定基座模型，请使用 --base 参数指定")
                    sys.exit(1)
        else:
            models = scan_for_models()
            if models:
                valid = [m for m in models if m["valid"]]
                if valid:
                    chosen = _interactive_select(
                        valid, "基座模型",
                        fmt_fn=lambda m: f"{m['name']} ({m.get('architecture', '?')}, {m.get('size', '?')})",
                    )
                    if chosen:
                        base_path = str(Path(chosen["path"]).resolve())
            if base_path is None:
                _error("未找到基座模型，请使用 --base 指定路径或 HF ID")
                sys.exit(1)

    # 校验 base_path
    bp = Path(base_path)
    if not bp.exists() and not args.base_only:
        # 可能是 HF ID，直接使用
        _info(f"基座模型 '{base_path}' 本地不存在，将尝试从 HuggingFace 在线加载")
    elif args.base_only and not bp.exists():
        _error(f"基座模型路径不存在: {base_path}")
        sys.exit(1)

    # ── 加载模型 ──
    _banner()

    cfg = InferConfig(
        base_path=base_path,
        adapter_path=adapter_path if not args.base_only else None,
        use_4bit=not args.no_4bit,
        backend=args.backend,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        seed=args.seed,
        verbose=not args.quiet,
    )

    engine = InferenceEngine(cfg)

    try:
        engine.load()
    except Exception as e:
        _error(f"模型加载失败: {e}")
        sys.exit(1)

    if args.system_prompt:
        engine.set_system_prompt(args.system_prompt)
        _info(f"系统提示: {args.system_prompt}")

    # ── 模式分发 ──

    # 单次问答（默认流式）
    if args.question:
        print(f"\n{C_BRIGHT}Q:{C_RESET} {args.question}")
        if args.no_stream:
            reply = engine.generate(args.question)
            print(f"{C_BRIGHT}A:{C_RESET} {reply}")
        else:
            print(f"{C_BRIGHT}A:{C_RESET} ", end="", flush=True)
            for chunk in engine.generate_stream(args.question):
                print(chunk, end="", flush=True)
            print()
        return

    # 测试模式（默认流式）
    if args.test:
        tests = [
            "What is the capital of France?",
            "If x + 3 = 7, what is x?",
            "Write a Python function to check if a number is prime.",
            "3.14和3.41哪个数值更高?",
        ]
        print(f"\n{C_CYAN}═══ 测试模式 ═══{C_RESET}\n")
        for q in tests:
            print(f"{C_BRIGHT}Q:{C_RESET} {q}")
            if args.no_stream:
                reply = engine.generate(q)
                print(f"{C_BRIGHT}A:{C_RESET} {reply}")
            else:
                print(f"{C_BRIGHT}A:{C_RESET} ", end="", flush=True)
                for chunk in engine.generate_stream(q):
                    print(chunk, end="", flush=True)
                print()
            print(f"{C_DIM}---{C_RESET}")
        return

    # 批量推理
    if args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            _error(f"批量推理文件不存在: {args.batch}")
            sys.exit(1)

        prompts = _load_batch_queries(batch_file)
        if not prompts:
            _error(f"未能从 {args.batch} 读取到有效问题")
            sys.exit(1)

        print(f"\n{C_CYAN}═══ 批量推理: {len(prompts)} 条 ═══{C_RESET}\n")
        for i, p in enumerate(prompts):
            print(f"\n{C_BRIGHT}[{i+1}/{len(prompts)}] Q:{C_RESET} {p}")
            if args.no_stream:
                reply = engine.generate(p)
                print(f"{C_BRIGHT}A:{C_RESET} {reply}")
            else:
                print(f"{C_BRIGHT}A:{C_RESET} ", end="", flush=True)
                for chunk in engine.generate_stream(p):
                    print(chunk, end="", flush=True)
                print()
        return

    # 多轮对话
    if args.chat:
        print(f"\n{C_CYAN}═══ 多轮对话模式{C_RESET}")
        print(f"  {C_DIM}输入 'clear' 清除历史, 'history' 查看历史, 'exit' 退出{C_RESET}\n")
        while True:
            try:
                q = input(f"{C_GREEN}>>> {C_RESET}")
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q.lower() == "clear":
                    engine.clear_history()
                    print(f"{C_YELLOW}对话历史已清除{C_RESET}")
                    continue
                if q.lower() == "history":
                    hist = engine.history
                    if not hist:
                        print(f"{C_DIM}(空){C_RESET}")
                    else:
                        for m in hist:
                            role = m["role"]
                            content = m["content"][:100]
                            color = C_CYAN if role == "user" else C_MAGENTA
                            print(f"{color}[{role}]{C_RESET} {content}...")
                    continue
                if not q.strip():
                    continue
                if args.no_stream:
                    reply = engine.chat(q)
                    print(f"{C_MAGENTA}{reply}{C_RESET}")
                else:
                    for chunk in engine.chat_stream(q):
                        print(chunk, end="", flush=True)
                    print()
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        return

    # 默认交互模式（默认流式）
    print(f"\n{C_CYAN}═══ 交互推理模式{C_RESET}")
    print(f"  {C_DIM}输入问题即可推理, 'exit' 退出{C_RESET}\n")
    while True:
        try:
            q = input(f"{C_GREEN}>>> {C_RESET}")
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q.strip():
                continue
            if args.no_stream:
                reply = engine.generate(q)
                print(f"{C_MAGENTA}{reply}{C_RESET}")
            else:
                for chunk in engine.generate_stream(q):
                    print(chunk, end="", flush=True)
                print()
            print()
        except (EOFError, KeyboardInterrupt):
            print()
            break

    # 清理
    engine.unload()
    print(f"\n{C_DIM}Bye.{C_RESET}")


# ── 批量推理辅助 ────────────────────────────────────────────

def _load_batch_queries(file_path: Path) -> list:
    """从文件加载批量问题列表"""
    if file_path.suffix == ".jsonl":
        queries = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("question") or item.get("prompt") or item.get("input") or item.get("instruction") or ""
                if q:
                    queries.append(str(q).strip())
        return queries
    elif file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            queries = []
            for item in data:
                if isinstance(item, str):
                    queries.append(item.strip())
                elif isinstance(item, dict):
                    q = item.get("question") or item.get("prompt") or item.get("input") or item.get("instruction") or ""
                    if q:
                        queries.append(str(q).strip())
            return queries
        elif isinstance(data, dict):
            # {"q1": "answer1", ...} 这种格式
            return list(data.keys())
    elif file_path.suffix in (".txt", ".csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines
    return []


if __name__ == "__main__":
    main()
