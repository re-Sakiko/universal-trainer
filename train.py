#!/usr/bin/env python
"""
Universal Trainer — CLI 训练入口
==================================

用法:
    python train.py --scan                    # 扫描 models/ + datasets/ + outputs/ 目录
    python train.py --validate                # 校验文件格式
    python train.py --dataset datasets/my_data.json
    python train.py --dataset data.csv --format csv

    # 单推理模式 (Standalone Inference)
    python train.py --infer-only                                 # 交互选择模型 + LoRA
    python train.py --infer-only --model outputs/my_lora         # 指定 LoRA
    python train.py --infer-only --model outputs/my_lora --stream
    python train.py --infer-only --model outputs/my_lora -q "问题"
    python train.py --infer-only --base-only --base models/Qwen2-0.5B
    python train.py --infer-only --model outputs/my_lora --chat
    python train.py --infer-only --model outputs/my_lora --batch qs.jsonl

    # 训练后自动推理
    python train.py --dataset data.json --model models/Qwen2-0.5B --infer
"""

import sys, argparse
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    from core.config import IS_ROCM
    HAS_ROCM = IS_ROCM
except Exception:
    HAS_CUDA = False
    HAS_ROCM = False

from core import (
    TrainConfig, TrainingEngine, load_dataset, list_supported_formats,
    scan_models, scan_datasets, print_scan_report, pick_model, pick_dataset,
    validate_all_models, validate_all_datasets, print_validation_report,
    InferenceEngine, InferConfig,
    resolve_base_from_adapter,
)


def _run_inference(args):
    """单推理模式入口"""
    from core.infer_engine import (
        scan_for_models as _scan_models,
        scan_for_adapters as _scan_adapters,
        IS_ROCM,
    )

    # ── 颜色 ──
    try:
        from colorama import init, Fore, Style
        init(autoreset=True)
        C = type("C", (), {
            "R": Fore.RED, "G": Fore.GREEN, "Y": Fore.YELLOW,
            "C": Fore.CYAN, "M": Fore.MAGENTA, "D": Style.DIM,
            "B": Style.BRIGHT, "X": Style.RESET_ALL,
        })
    except ImportError:
        C = type("C", (), {k: "" for k in "RGYCMDBX"})

    def _ok(s):  print(f"{C.G}✓ {s}{C.X}")
    def _err(s): print(f"{C.R}✗ {s}{C.X}", file=sys.stderr)

    # 解析路径
    adapter_path = args.model if not args.base_only else None
    base_path = args.base

    # 扫描
    if adapter_path is None and not args.base_only:
        adapters = _scan_adapters()
        if adapters:
            if len(adapters) == 1:
                adapter_path = str(Path(adapters[0]["path"]).resolve())
                _ok(f"自动选择 LoRA: {adapters[0]['name']}")
            else:
                print(f"\n发现 {len(adapters)} 个 LoRA adapter:")
                for i, a in enumerate(adapters):
                    print(f"  [{i+1}] {a['name']} (rank={a.get('rank','?')})")
                try:
                    c = input(f"选择 [1-{len(adapters)}]: ").strip()
                    idx = int(c) - 1
                    if 0 <= idx < len(adapters):
                        adapter_path = str(Path(adapters[idx]["path"]).resolve())
                except (ValueError, EOFError):
                    pass

    # 校验 adapter
    if adapter_path:
        p = Path(adapter_path)
        if not p.exists() or not (p / "adapter_config.json").exists():
            _err(f"无效的 LoRA adapter: {adapter_path}")
            sys.exit(1)

    # 解析基座
    if base_path is None:
        if adapter_path:
            detected = resolve_base_from_adapter(adapter_path)
            if detected:
                bp = Path(detected)
                if bp.exists():
                    base_path = str(bp.resolve())
                    _ok(f"检测到基座模型: {base_path}")
                else:
                    base_path = detected
                    _ok(f"检测到基座模型: {detected} (在线)")
        if base_path is None:
            models = _scan_models()
            valid = [m for m in models if m["valid"]]
            if valid:
                if len(valid) == 1:
                    base_path = str(Path(valid[0]["path"]).resolve())
                    _ok(f"自动选择基座: {valid[0]['name']}")
                else:
                    print(f"\n发现 {len(valid)} 个基座模型:")
                    for i, m in enumerate(valid):
                        print(f"  [{i+1}] {m['name']} ({m.get('architecture','?')})")
                    try:
                        c = input(f"选择 [1-{len(valid)}]: ").strip()
                        idx = int(c) - 1
                        if 0 <= idx < len(valid):
                            base_path = str(Path(valid[idx]["path"]).resolve())
                    except (ValueError, EOFError):
                        pass
        if base_path is None:
            _err("无法确定基座模型，请使用 --base 指定")
            sys.exit(1)

    # 配置 & 加载
    cfg = InferConfig(
        base_path=base_path,
        adapter_path=adapter_path if not args.base_only else None,
        use_4bit=not args.no_4bit,
        backend=args.backend,
        max_new_tokens=args.infer_max_tokens or args.max_tokens or 512,
        temperature=args.infer_temperature or args.temperature or 0.7,
        top_p=args.infer_top_p or args.top_p or 0.9,
        top_k=args.infer_top_k or args.top_k or 50,
        repetition_penalty=args.repetition_penalty or 1.1,
        do_sample=not args.no_sample,
        seed=args.seed or 42,
        verbose=not args.quiet,
    )

    print(f"\n加载模型...")
    print(f"  基座: {base_path}")
    if adapter_path and not args.base_only:
        print(f"  LoRA: {adapter_path}")

    engine = InferenceEngine(cfg)
    try:
        engine.load()
    except Exception as e:
        _err(f"模型加载失败: {e}")
        sys.exit(1)

    if args.system_prompt:
        engine.set_system_prompt(args.system_prompt)

    # 分发
    if args.question:
        if not args.no_stream:
            print(f"\nQ: {args.question}")
            print(f"A: ", end="", flush=True)
            for chunk in engine.generate_stream(args.question):
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"\nQ: {args.question}")
            print(f"A: {engine.generate(args.question)}")
    elif args.test:
        tests = [
            "What is the capital of France?",
            "If x + 3 = 7, what is x?",
            "3.14和3.41哪个数值更高?",
        ]
        print(f"\n═══ 测试 ═══\n")
        for q in tests:
            print(f"Q: {q}")
            if not args.no_stream:
                print(f"A: ", end="", flush=True)
                for chunk in engine.generate_stream(q):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(f"A: {engine.generate(q)}")
            print("---")
    elif args.chat:
        print(f"\n═══ 多轮对话\n输入 'clear' 清除历史, 'exit' 退出\n")
        while True:
            try:
                q = input(">>> ")
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q.lower() == "clear":
                    engine.clear_history()
                    print("历史已清除")
                    continue
                if not q.strip():
                    continue
                if not args.no_stream:
                    for chunk in engine.chat_stream(q):
                        print(chunk, end="", flush=True)
                    print()
                else:
                    print(engine.chat(q))
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break
    elif args.infer_batch:
        import json
        p = Path(args.infer_batch)
        if not p.exists():
            _err(f"批量文件不存在: {args.infer_batch}")
            sys.exit(1)
        prompts = []
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        q = item.get("question") or item.get("prompt") or ""
                        if q: prompts.append(str(q))
        elif p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    prompts.append(str(item) if isinstance(item, str) else item.get("question", item.get("prompt", "")))
        elif p.suffix in (".txt", ".csv"):
            with open(p, encoding="utf-8") as f:
                prompts = [l.strip() for l in f if l.strip()]

        if not prompts:
            _err("未能读取到问题")
            sys.exit(1)

        print(f"\n═══ 批量推理: {len(prompts)} 条 ═══\n")
        for i, q in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Q: {q}")
            if not args.no_stream:
                print(f"A: ", end="", flush=True)
                for chunk in engine.generate_stream(q):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(f"A: {engine.generate(q)}")
    else:
        print(f"\n═══ 交互模式\n输入问题, 'exit' 退出\n")
        while True:
            try:
                q = input(">>> ")
                if q.lower() in ("exit", "quit", "q"):
                    break
                if not q.strip():
                    continue
                if not args.no_stream:
                    for chunk in engine.generate_stream(q):
                        print(chunk, end="", flush=True)
                    print()
                else:
                    print(engine.generate(q))
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break

    engine.unload()


def main():
    parser = argparse.ArgumentParser(description="Universal Trainer — 通用大模型微调框架")
    parser.add_argument("--scan", action="store_true", help="扫描 models/ datasets/ outputs/ 目录")
    parser.add_argument("--validate", action="store_true", help="扫描并校验 models/ 和 datasets/ 格式")

    # 数据集
    parser.add_argument("--dataset", type=str, help="数据集路径 (跳过扫描)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "sharegpt", "alpaca", "jsonl", "csv", "text"])

    # 模型
    parser.add_argument("--model", type=str, help="基座模型或LoRA路径")
    parser.add_argument("--base", type=str, default=None, help="基座模型路径 (推理时用)")
    parser.add_argument("--base-only", action="store_true", help="纯base推理，不加载LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="从已有 LoRA 继续训练")

    # 训练
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=None)

    # 后端
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "directml", "mps", "cpu"])
    parser.add_argument("--no-4bit", action="store_true", help="禁用4-bit量化")

    # 输出
    parser.add_argument("--output", type=str, default="outputs/trained_model")

    # ── 推理模式 ──
    infer_group = parser.add_argument_group("推理模式 (Standalone Inference)")
    infer_group.add_argument("--infer-only", action="store_true",
                             help="跳过训练，直接进入推理模式")
    infer_group.add_argument("--infer", action="store_true",
                             help="训练完成后自动进行推理测试")
    infer_group.add_argument("--no-stream", action="store_true",
                             help="禁用流式逐字输出（默认开启）")
    infer_group.add_argument("--chat", action="store_true",
                             help="多轮对话模式")
    infer_group.add_argument("-q", "--question", type=str, default=None,
                             help="单次问答")
    infer_group.add_argument("--test", action="store_true",
                             help="运行内置推理测试")
    infer_group.add_argument("--infer-batch", type=str, default=None, metavar="FILE",
                             help="批量推理文件 (JSONL/JSON/TXT)")
    infer_group.add_argument("--system-prompt", type=str, default=None,
                             help="设置系统提示词")

    # ── 生成参数 ──
    gen_group = parser.add_argument_group("生成参数")
    gen_group.add_argument("--max-tokens", type=int, default=None,
                           help="最大生成 token 数")
    gen_group.add_argument("--temperature", type=float, default=None,
                           help="采样温度")
    gen_group.add_argument("--top-p", type=float, default=None,
                           help="nucleus sampling")
    gen_group.add_argument("--top-k", type=int, default=None,
                           help="top-k sampling")
    gen_group.add_argument("--repetition-penalty", type=float, default=None,
                           help="重复惩罚")
    gen_group.add_argument("--no-sample", action="store_true",
                           help="禁用采样 (贪心解码)")
    gen_group.add_argument("--seed", type=int, default=None,
                           help="随机种子")
    gen_group.add_argument("--quiet", action="store_true",
                           help="静默模式")

    # ── 推理参数别名 (兼容 infer.py 的参数名) ──
    infer_alias = parser.add_argument_group("推理参数别名")
    infer_alias.add_argument("--infer-max-tokens", type=int, default=None, help=argparse.SUPPRESS)
    infer_alias.add_argument("--infer-temperature", type=float, default=None, help=argparse.SUPPRESS)
    infer_alias.add_argument("--infer-top-p", type=float, default=None, help=argparse.SUPPRESS)
    infer_alias.add_argument("--infer-top-k", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── 单推理模式 ──
    if args.infer_only:
        _run_inference(args)
        return

    # ---- 扫描模式 ----
    if args.scan:
        print_scan_report()
        return

    # ---- 校验模式 ----
    if args.validate:
        print_validation_report()
        return

    # ---- 确定模型路径 ----
    model_path = args.model
    if not model_path:
        model_path = pick_model()
        if not model_path:
            print("\n请指定模型路径: --model models/你的模型文件夹")
            print("或运行: python train.py --scan 查看可用资源")
            sys.exit(1)

    # ---- 确定数据集路径 ----
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = pick_dataset()
        if not dataset_path:
            print("\n请指定数据集: --dataset datasets/你的数据.json")
            print("或运行: python train.py --scan 查看可用资源")
            sys.exit(1)

    # ---- 训练 ----
    print("\n" + "=" * 50)
    print("  Universal Trainer")
    print("=" * 50)
    print(f"  模型: {model_path}")
    print(f"  数据: {dataset_path}")
    print(f"  格式: {args.format}")
    print(f"  后端: {args.backend}")
    if HAS_ROCM:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            hip_ver = torch.version.hip
            print(f"  GPU: {gpu_name} (HIP {hip_ver})")
        except Exception:
            pass
    elif HAS_CUDA:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
        except Exception:
            pass
    print(f"  步数: {args.max_steps}")
    print(f"  输出: {args.output}")
    print("=" * 50 + "\n")

    cfg = TrainConfig(
        dataset_path=dataset_path,
        dataset_format=args.format,
        model_path=model_path,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size or (2 if (HAS_CUDA or HAS_ROCM) else 1),
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length or 1024,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output,
        backend=args.backend,
        resume=args.resume,
    )

    engine = TrainingEngine(cfg)
    engine.prepare(resume=args.resume)
    engine.train()

    # ---- 推理测试 ----
    if args.infer or args.question or args.test:
        print("\n" + "=" * 50)
        print("  推理测试")
        print("=" * 50)

        if args.question:
            print(f"\nQ: {args.question}")
            print(f"A: {engine.infer(args.question)}")
        elif args.test:
            tests = [
                "What is the capital of France?",
                "If x + 3 = 7, what is x?",
            ]
            for q in tests:
                print(f"\nQ: {q}")
                print(f"A: {engine.infer(q)}")
        else:
            tests = ["What is the capital of France?", "If x + 3 = 7, what is x?"]
            for q in tests:
                print(f"\nQ: {q}")
                print(f"A: {engine.infer(q)}")


if __name__ == "__main__":
    main()
