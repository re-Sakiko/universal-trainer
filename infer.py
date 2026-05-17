#!/usr/bin/env python
"""
推理入口
========
用法:
    python infer.py                              # 自动扫描 models/ 和 outputs/
    python infer.py --model outputs/trained_model # 使用指定 LoRA，自动检测基座模型
    python infer.py --model outputs/trained_model --base models/your-model
    python infer.py --model outputs/trained_model --test
    python infer.py --model outputs/trained_model -q "问题"
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import torch

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from core.scanner import scan_models, scan_outputs


def _detect_rocm() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True
        name = torch.cuda.get_device_name(0)
        if any(x in name for x in ("AMD", "Radeon", "Instinct", "gfx")):
            return True
    except Exception:
        pass
    return False


def _resolve_base_from_adapter(adapter_path: str) -> Optional[str]:
    """
    从 LoRA adapter 的 adapter_config.json 中读取 base_model_name_or_path。
    返回本地路径或 HF 模型 ID；找不到时返回 None。
    """
    adapter_cfg = Path(adapter_path) / "adapter_config.json"
    if not adapter_cfg.exists():
        return None
    try:
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            return None
        # 如果原始路径已存在，直接使用
        bp = Path(base_name)
        if bp.exists():
            return str(bp.resolve())
        # 在 models/ 下按名字匹配
        for m in scan_models():
            if m["name"] == bp.name or m["name"] == base_name or m["path"] == base_name:
                return m["path"]
        # 未找到本地匹配，返回原始值（可能是 HF 模型 ID）
        return base_name
    except Exception:
        return None


def _interactive_select(items: list, label: str, fmt_fn=None) -> Optional[dict]:
    """通用交互式选择：单项自动确认，多项展示编号列表让用户选择。"""
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    print(f"\n发现 {len(items)} 个{label}:")
    for i, item in enumerate(items):
        desc = fmt_fn(item) if fmt_fn else str(item.get("name", item))
        print(f"  [{i + 1}] {desc}")
    while True:
        try:
            choice = input(f"选择{label} [1-{len(items)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"请输入 1-{len(items)} 之间的数字")
        except (ValueError, IndexError, EOFError):
            print("输入无效，请重试")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="LoRA 模型路径")
    parser.add_argument("--base", type=str, default=None, help="基座模型")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-q", "--question", type=str)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    HAS_GPU = torch.cuda.is_available() and not args.no_cuda
    IS_ROCM = _detect_rocm() if HAS_GPU else False

    # ============================================================
    # 1. 解析 LoRA adapter 路径 (--model)
    # ============================================================
    model_path: Optional[str] = args.model

    if model_path is not None:
        # 用户明确指定了路径 -> 校验是否存在
        p = Path(model_path)
        if not p.exists():
            print(f"错误: 指定的 LoRA 路径不存在: {model_path}")
            outputs = scan_outputs()
            if outputs:
                print(f"  outputs/ 目录下可用的 LoRA adapter:")
                for o in outputs:
                    print(f"    - {o['path']}")
            else:
                print(f"  outputs/ 目录下没有找到 LoRA adapter")
                print(f"  请先训练模型或将 LoRA 文件夹放入 outputs/ 目录")
            sys.exit(1)
        if not (p / "adapter_config.json").exists():
            print(f"错误: {model_path} 不是有效的 LoRA adapter（缺少 adapter_config.json）")
            sys.exit(1)
    else:
        # 自动扫描 outputs/
        outputs = scan_outputs()
        if not outputs:
            print("错误: 未指定 --model，且 outputs/ 目录下没有找到 LoRA adapter")
            print(f"  请先训练模型或将 LoRA 文件夹放入 outputs/ 目录")
            sys.exit(1)
        chosen = _interactive_select(
            outputs, "LoRA adapter",
            fmt_fn=lambda x: x["name"],
        )
        if chosen is None:
            print("错误: 未选择 LoRA adapter")
            sys.exit(1)
        model_path = str(Path(chosen["path"]).resolve())
        print(f"已选择 LoRA adapter: {chosen['name']}")

    # ============================================================
    # 2. 解析基座模型路径 (--base)
    # ============================================================
    base_path: Optional[str] = args.base

    if base_path is not None:
        # 用户明确指定了 --base -> 直接使用
        p = Path(base_path)
        if not p.exists():
            print(f"警告: 指定的基座模型路径不存在: {base_path}")
            avail = scan_models()
            if avail:
                print(f"  models/ 目录下可用的模型:")
                for m in avail:
                    status = "valid" if m["valid"] else "invalid"
                    print(f"    - {m['path']}  ({status})")
            print(f"  将尝试作为 HuggingFace 模型 ID 在线加载")
    else:
        # 2a. 尝试从 LoRA adapter 配置中自动检测
        detected = _resolve_base_from_adapter(model_path)
        if detected is not None:
            bp = Path(detected)
            if bp.exists():
                base_path = str(bp.resolve())
                print(f"从 LoRA adapter 配置中检测到基座模型: {base_path}")
            else:
                # 可能是 HF 模型 ID，仍然使用
                base_path = detected
                print(f"从 LoRA adapter 配置中检测到基座模型: {detected}")
                print(f"  (本地未找到，将尝试在线加载；可指定 --base 使用本地模型)")
        else:
            # 2b. 扫描 models/ 目录
            models = scan_models()
            if not models:
                print("错误: 未指定 --base，且 models/ 目录下没有找到模型")
                print(f"  请将 HuggingFace 模型文件夹放入 models/ 目录")
                sys.exit(1)

            valid = [m for m in models if m["valid"]]
            if not valid:
                print("错误: models/ 目录下找到的文件夹格式都不正确:")
                for m in models:
                    errs = ", ".join(m["errors"]) if m["errors"] else "未知错误"
                    print(f"  FAIL {m['name']}: {errs}")
                sys.exit(1)

            chosen = _interactive_select(
                valid, "基座模型",
                fmt_fn=lambda m: f"{m['name']} ({m['architecture']}, {m['size']})",
            )
            if chosen is None:
                print("错误: 未选择基座模型")
                sys.exit(1)
            base_path = str(Path(chosen["path"]).resolve())
            print(f"已选择基座模型: {chosen['name']} ({chosen['architecture']}, {chosen['size']})")

    # ============================================================
    # 3. 加载模型
    # ============================================================
    gpu_label = (
        f"ROCm (HIP {torch.version.hip})"
        if IS_ROCM
        else "CUDA" if HAS_GPU else None
    )
    load_label = gpu_label or "CPU"
    print(f"加载基座模型: {base_path}")
    print(f"加载 LoRA: {model_path} [{load_label}]")

    if HAS_GPU:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            dtype=torch.float32,
        )

    # Tokenizer: 优先从 LoRA adapter 目录加载；如果 LoRA 没有 tokenizer 则从基座模型加载
    tokenizer_path = model_path
    if not (Path(model_path) / "tokenizer_config.json").exists():
        tokenizer_path = base_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    model.config.use_cache = True

    # BF16 fix (避免 lm_head 因 4bit 量化导致的 dtype 不匹配)
    if HAS_GPU:
        try:
            base = model.base_model.model if hasattr(model, "base_model") else model
            if hasattr(base, "lm_head"):
                base.lm_head = base.lm_head.to(torch.bfloat16)
        except Exception:
            pass

    # ============================================================
    # 4. 推理
    # ============================================================
    def ask(q, max_tok=256):
        msgs = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        if HAS_GPU:
            inputs = inputs.to("cuda")
        with torch.no_grad():
            if HAS_GPU:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tok,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        marker = "<|im_start|>assistant\n"
        if marker in decoded:
            return decoded.split(marker)[-1].split("<|im_end|>")[0].strip()
        return decoded.strip()

    if args.question:
        print(f"Q: {args.question}")
        print(f"A: {ask(args.question)}")
        return

    if args.test:
        tests = [
            "What is the capital of France?",
            "If x + 3 = 7, what is x?",
            "3.14和3.41哪个数值更高?",
        ]
        for q in tests:
            print(f"Q: {q}")
            print(f"A: {ask(q)}")
            print("---")
        return

    print("\n输入问题，输入 exit 退出\n")
    while True:
        try:
            q = input(">>> ")
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q.strip():
                continue
            print(ask(q))
            print()
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    main()
