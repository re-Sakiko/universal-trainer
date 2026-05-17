#!/usr/bin/env python
"""
Universal Trainer — CLI 训练入口
==================================

用法:
    python train.py --scan                    # 扫描 models/ + datasets/ + outputs/ 目录
    python train.py --validate                # 校验文件格式
    python train.py --dataset datasets/my_data.json
    python train.py --dataset data.csv --format csv
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
)


def main():
    parser = argparse.ArgumentParser(description="Universal Trainer — 通用大模型微调框架")
    parser.add_argument("--scan", action="store_true", help="扫描 models/ datasets/ outputs/ 目录")
    parser.add_argument("--validate", action="store_true", help="扫描并校验 models/ 和 datasets/ 格式")

    # 数据集
    parser.add_argument("--dataset", type=str, help="数据集路径 (跳过扫描)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "sharegpt", "alpaca", "jsonl", "csv", "text"])

    # 模型
    parser.add_argument("--model", type=str, help="基座模型路径 (跳过扫描)")
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

    # 输出
    parser.add_argument("--output", type=str, default="outputs/trained_model")

    args = parser.parse_args()

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
    print("\n" + "=" * 50)
    print("  推理测试")
    print("=" * 50)
    tests = ["What is the capital of France?", "If x + 3 = 7, what is x?"]
    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {engine.infer(q)}")


if __name__ == "__main__":
    main()
