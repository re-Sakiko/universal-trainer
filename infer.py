#!/usr/bin/env python
"""
推理入口
========
用法:
    python infer.py --model outputs/trained_model
    python infer.py --model outputs/trained_model --test
    python infer.py --model outputs/trained_model -q "问题"
"""

import sys, argparse
from pathlib import Path
import torch

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/trained_model", help="LoRA 模型路径")
    parser.add_argument("--base", type=str, default="models/Qwen2-0.5B-Instruct", help="基座模型")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-q", "--question", type=str)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    HAS_GPU = torch.cuda.is_available() and not args.no_cuda
    IS_ROCM = _detect_rocm() if HAS_GPU else False

    # 加载
    gpu_label = f"ROCm (HIP {torch.version.hip})" if IS_ROCM else "CUDA" if HAS_GPU else None
    load_label = gpu_label or "CPU"
    print(f"加载模型: {args.model} [{load_label}]")

    if HAS_GPU:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, trust_remote_code=True, torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.model)
    model.eval()
    model.config.use_cache = True

    # BF16 fix
    if HAS_GPU:
        try:
            base = model.base_model.model if hasattr(model, "base_model") else model
            if hasattr(base, "lm_head"):
                base.lm_head = base.lm_head.to(torch.bfloat16)
        except Exception:
            pass

    def ask(q, max_tok=256):
        msgs = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        if HAS_GPU:
            inputs = inputs.to("cuda")
        with torch.no_grad():
            if HAS_GPU:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            else:
                outputs = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        marker = "<|im_start|>assistant\n"
        return decoded.split(marker)[-1].split("<|im_end|>")[0].strip() if marker in decoded else decoded.strip()

    if args.question:
        print(f"Q: {args.question}")
        print(f"A: {ask(args.question)}")
        return

    if args.test:
        tests = ["What is the capital of France?", "If x + 3 = 7, what is x?", "3.14和3.41哪个数值更高?"]
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
