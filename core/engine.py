"""
多后端训练引擎
===============
支持 CUDA / DirectML / MPS / CPU
自动检测硬件 → 选择最佳配置 → 训练 → 保存
"""

import os, sys, time, threading
from pathlib import Path
from typing import Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from .config import TrainConfig, IS_ROCM
from .dataset_loader import load_dataset

BASE = Path(__file__).parent.parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class TrainingEngine:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self._stop_flag = False
        self._log_callback: Optional[Callable] = None
        self._log_lines: list = []

    def set_log_callback(self, cb: Callable):
        self._log_callback = cb

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if self._log_callback:
            self._log_callback(line)
        print(msg)

    def _get_device(self):
        cfg = self.config
        if cfg.backend in ("cuda", "rocm"):
            return torch.device("cuda")
        elif cfg.backend == "directml":
            import torch_directml
            return torch_directml.device()
        elif cfg.backend == "mps":
            return torch.device("mps")
        return torch.device("cpu")

    def _start_stop_listener(self):
        """启动后台线程监听 Ctrl+F 终止信号"""
        self._stop_flag = False
        t = threading.Thread(target=self._keyboard_listener, daemon=True)
        t.start()

    def _keyboard_listener(self):
        """跨平台键盘监听：检测 Ctrl+F (ASCII 0x06)"""
        try:
            # Windows: 使用 msvcrt
            import msvcrt
            while not self._stop_flag:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'\x06':  # Ctrl+F
                        self._log("收到 Ctrl+F 停止信号")
                        self._stop_flag = True
                        break
                else:
                    time.sleep(0.1)
        except ImportError:
            # Unix: 使用 select + sys.stdin
            import select, termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop_flag:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == '\x06':  # Ctrl+F
                            self._log("收到 Ctrl+F 停止信号")
                            self._stop_flag = True
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def prepare(self, resume: bool = False):
        """加载模型 + 数据集，准备训练"""
        cfg = self.config
        resume = resume or cfg.resume
        backend_label = "ROCm" if cfg.is_rocm else cfg.backend.upper()
        self._log(f"后端: {backend_label}")
        self._log(f"精度: {cfg.dtype}, 4-bit: {cfg.use_4bit}")

        # ROCm: 检测可用的 attention 实现
        if cfg.is_rocm:
            self._rocm_attn = _detect_rocm_attn()
            self._log(f"ROCm Attention: {self._rocm_attn}")

        # ---- 加载数据集 ----
        self._log(f"加载数据集: {cfg.dataset_path}")
        self.dataset = load_dataset(
            cfg.dataset_path,
            format=cfg.dataset_format,
            max_samples=cfg.max_samples,
        )

        # ---- 加载模型 ----
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {cfg.model_path}")

        self._log(f"加载模型: {cfg.model_name}")

        load_kwargs = dict(trust_remote_code=True, dtype=cfg.torch_dtype)

        # ROCm: 设置 attention 实现
        if cfg.is_rocm and self._rocm_attn != "sdpa":
            load_kwargs["attn_implementation"] = self._rocm_attn

        if cfg.use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=cfg.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)

        # 非 4-bit: 手动移动设备
        if not cfg.use_4bit and cfg.backend in ("directml", "mps"):
            self.model = self.model.to(self._get_device())

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- LoRA: 新训练 或 继续训练 ----
        if cfg.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        if cfg.backend in ("cuda", "rocm"):
            self.model.gradient_checkpointing_enable()

        # ROCm: 尝试 torch.compile 加速
        if cfg.is_rocm and _can_compile():
            try:
                self._log("ROCm: 启用 torch.compile...")
                self.model = torch.compile(self.model, backend="inductor", mode="reduce-overhead")
            except Exception as e:
                self._log(f"ROCm: torch.compile 跳过 ({e})")

        if resume:
            from peft import PeftModel
            adapter_path = Path(cfg.output_dir)
            if not (adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"未找到已有 LoRA 适配器: {cfg.output_dir}\n"
                    f"请确认路径正确，或去掉 --resume 重新训练"
                )
            self._log(f"继续训练: 加载已有 LoRA 适配器 {cfg.output_dir}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=True)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self._log(f"LoRA (继续): {trainable:,} / {total:,} 可训练 ({100*trainable/total:.2f}%)")
        else:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.target_modules,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self._log(f"LoRA: {trainable:,} / {total:,} 可训练 ({100*trainable/total:.2f}%)")

        # ---- 格式化数据集 ----
        self._log("格式化数据集...")
        tok = self.tokenizer

        def fmt(example, _tok):
            text = _tok.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        self.dataset = self.dataset.map(
            fmt, fn_kwargs={"_tok": tok}, num_proc=1, remove_columns=["messages"]
        )

    def train(self):
        """开始训练"""
        cfg = self.config

        sft_config = SFTConfig(
            output_dir=str(Path(cfg.output_dir).parent / "checkpoints"),
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            fp16=(cfg.dtype == "float16"),
            bf16=cfg.use_bf16,
            logging_steps=max(1, cfg.max_steps // 20) if cfg.max_steps >= 20 else 1,
            optim="adamw_8bit" if cfg.backend in ("cuda", "rocm") else "adamw_torch",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="linear",
            seed=cfg.seed,
            report_to="none",
            save_strategy="no",
            dataloader_num_workers=0,
            gradient_checkpointing=(cfg.backend in ("cuda", "rocm")),
            max_length=cfg.max_seq_length,
            dataset_num_proc=1,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
            args=sft_config,
        )

        self._log(f"开始训练: {cfg.max_steps} 步, batch={cfg.effective_batch_size}")
        self._log(f"配置: lr={cfg.learning_rate}, LoRA r={cfg.lora_r}, seq={cfg.max_seq_length}")
        self._log("按 Ctrl+F 可随时终止训练并自动保存模型")

        # 启动键盘监听线程
        self._start_stop_listener()

        # 拦截 logging
        original_log = self.trainer.log
        def custom_log(logs, start_time=None):
            if not self._stop_flag:
                loss = logs.get("loss", "?")
                step = logs.get("step", "?")
                lr = logs.get("learning_rate", "?")
                self._log(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}" if isinstance(loss, float) else f"Step {step}: loss={loss}")
            if not self._stop_flag:
                return original_log(logs, start_time)
        self.trainer.log = custom_log

        try:
            self.trainer.train()
        except KeyboardInterrupt:
            self._log("训练被 Ctrl+C 中断")

        self.save()

    def stop(self):
        self._stop_flag = True

    def save(self):
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output))
        self.tokenizer.save_pretrained(str(output))
        self._log(f"模型已保存: {output}")

    def infer(self, question: str, max_tokens: int = 256) -> str:
        """单次推理"""
        if self.model is None or self.tokenizer is None:
            return "模型未加载"

        self.model.eval()
        self.model.config.use_cache = True

        # BF16 fix
        if self.config.backend in ("cuda", "rocm"):
            try:
                base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
                if hasattr(base, "lm_head"):
                    base.lm_head = base.lm_head.to(torch.bfloat16)
            except Exception:
                pass

        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        if self.config.backend in ("cuda", "rocm"):
            inputs = inputs.to("cuda")

        # 收集停止 token：eos + eot，优先从 generation_config 获取
        stop_ids = []
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            gc_eos = getattr(self.model.generation_config, 'eos_token_id', None)
            if gc_eos is not None:
                if isinstance(gc_eos, list):
                    stop_ids.extend(gc_eos)
                elif isinstance(gc_eos, int):
                    stop_ids.append(gc_eos)
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is not None:
                if isinstance(val, list):
                    stop_ids.extend(val)
                elif isinstance(val, int):
                    stop_ids.append(val)
        seen = set()
        stop_ids = [x for x in stop_ids if not (x in seen or seen.add(x))]

        with torch.no_grad():
            if self.config.backend in ("cuda", "rocm"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=stop_ids,
                    )
            else:
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=stop_ids,
                )

        input_len = inputs["input_ids"].shape[1]
        new_token_ids = outputs[0][input_len:]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    def get_logs(self) -> list:
        return self._log_lines


def _detect_rocm_attn() -> str:
    """检测 ROCm 可用的最优 attention 实现"""
    try:
        # flash_attention_2 在 ROCm 5.7+ / PyTorch 2.5+ 可用
        import flash_attn
        _ = flash_attn
        return "flash_attention_2"
    except ImportError:
        pass
    try:
        # SDPA 通常是可靠的回退
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"
    except Exception:
        pass
    return "eager"


def _can_compile() -> bool:
    """检测 torch.compile 是否可用"""
    try:
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) < (2, 5):
            return False
        # 快速 smoke test
        @torch.compile(backend="inductor")
        def _f(x):
            return x + 1
        _f(torch.tensor([1.0]))
        return True
    except Exception:
        return False
