"""训练配置"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import torch


@dataclass
class TrainConfig:
    # 模型
    model_name: str = "Qwen2-0.5B-Instruct"
    model_path: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

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
    backend: str = "auto"  # auto, cuda, directml, mps, cpu

    # 输出
    output_dir: str = "outputs/trained_model"

    # 内部状态
    use_4bit: bool = True
    use_bf16: bool = True
    dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.model_path:
            self.model_path = f"models/{self.model_name}"

        # 后端自动配置
        if self.backend == "auto":
            if torch.cuda.is_available():
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
