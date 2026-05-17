"""训练配置"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path
import torch


def _detect_rocm() -> bool:
    """检测是否为 ROCm (AMD GPU) 环境"""
    if not torch.cuda.is_available():
        return False
    try:
        # ROCm PyTorch 构建会设置 torch.version.hip
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True
        # 回退: 检查 GPU 厂商名
        name = torch.cuda.get_device_name(0)
        if any(x in name for x in ("AMD", "Radeon", "Instinct", "gfx")):
            return True
    except Exception:
        pass
    return False


IS_ROCM = _detect_rocm()


class ROCmConfig:
    """ROCm 特定优化配置"""
    attn_implementation: str = "auto"  # auto → 优先 flash_attention_2
    tunableop_enabled: bool = True     # PYTORCH_TUNABLEOP_ENABLED
    tf32_enabled: bool = False         # ROCm 上 TF32 精度不稳定，建议关闭
    conv_algos: str = "0"             # MI200+ 关闭 TF32 卷积


@dataclass
class TrainConfig:
    # 模型
    model_name: str = ""
    model_path: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Union[List[str], str] = "all-linear"

    # 数据
    dataset_path: str = ""
    dataset_format: str = "auto"  # auto, sharegpt, alpaca, jsonl, csv, text
    max_samples: Optional[int] = None
    max_seq_length: int = 1024

    # 训练超参
    max_steps: int = 2000
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 20
    weight_decay: float = 0.01
    seed: int = 3407

    # 后端
    backend: str = "auto"  # auto, cuda, rocm, directml, mps, cpu

    # 输出
    output_dir: str = "outputs/trained_model"

    # 内部状态
    use_4bit: bool = True
    use_bf16: bool = True
    dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.model_path and self.model_name:
            self.model_path = f"models/{self.model_name}"

        # 后端自动配置
        if self.backend == "auto":
            if torch.cuda.is_available():
                if IS_ROCM:
                    self.backend = "rocm"
                else:
                    self.backend = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.backend = "mps"
            else:
                try:
                    import torch_directml
                    torch_directml.device()
                    self.backend = "directml"
                except Exception:
                    self.backend = "cpu"

        # ROCm: 等同于 CUDA 的能力 + AMD 特定优化
        if self.backend == "rocm":
            self.use_4bit = True
            self.use_bf16 = True
            self.dtype = "bfloat16"
            _apply_rocm_env()

        # 按后端调整默认值
        if self.backend == "cpu":
            self.use_4bit = False
            self.use_bf16 = False
            self.dtype = "float32"
            self.per_device_batch_size = min(self.per_device_batch_size, 1)
            self.max_seq_length = min(self.max_seq_length, 512)
        elif self.backend in ("directml", "mps"):
            self.use_4bit = False
            self.use_bf16 = False
            self.dtype = "float16"

    @property
    def effective_batch_size(self):
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def torch_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

    @property
    def is_rocm(self) -> bool:
        return self.backend == "rocm"


def _apply_rocm_env():
    """应用 ROCm 环境变量优化"""
    cfg = ROCmConfig()
    if cfg.tunableop_enabled:
        os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
    if not cfg.tf32_enabled:
        os.environ.setdefault("TORCH_BACKEND_CUDA_ENABLE_TF32", "0")
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
