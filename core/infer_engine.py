"""
独立推理引擎
============
支持 base model / LoRA adapter 推理，多后端，流式输出，批量推理，多轮对话。

用法:
    from core.infer_engine import InferenceEngine, InferConfig

    cfg = InferConfig(base_path="models/Qwen2-0.5B", adapter_path="outputs/my_lora")
    engine = InferenceEngine(cfg)
    engine.load()

    # 单次推理
    answer = engine.generate("What is AI?")

    # 流式推理
    for chunk in engine.generate_stream("Explain quantum computing"):
        print(chunk, end="", flush=True)

    # 多轮对话
    engine.chat("Hi")
    engine.chat("Tell me more")

    # 批量推理
    results = engine.generate_batch(["Q1", "Q2", "Q3"])
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
)
from threading import Thread

BASE = Path(__file__).parent.parent


# ── 配置 ────────────────────────────────────────────────────

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


IS_ROCM = _detect_rocm()


@dataclass
class InferConfig:
    """推理配置"""
    # 模型路径
    base_path: str = ""                                # 基座模型路径 (HF ID 或本地路径)
    adapter_path: Optional[str] = None                 # LoRA adapter 路径 (None = 纯 base 推理)
    
    # 加载参数
    use_4bit: bool = True                              # 4-bit 量化
    device_map: str = "auto"
    trust_remote_code: bool = True
    backend: str = "auto"                              # auto / cuda / rocm / cpu
    
    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # 系统设置
    seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.backend == "auto":
            if torch.cuda.is_available():
                self.backend = "rocm" if IS_ROCM else "cuda"
            else:
                self.backend = "cpu"
        if self.backend == "cpu":
            self.use_4bit = False


# ── ChatMessage 辅助 ────────────────────────────────────────

def _extract_new_tokens(tokenizer, outputs, inputs) -> str:
    """仅解码模型新生成的 token（跳过输入 prompt），不依赖特定格式标记。

    这避免了硬编码特定模型（Qwen / Gemma / LLaMA）的对话标记。
    """
    input_len = inputs["input_ids"].shape[1]
    new_token_ids = outputs[0][input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def _build_chat_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """构建对话 prompt，兼容不支持 chat_template 的 tokenizer"""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # 手动回退
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ── 引擎 ────────────────────────────────────────────────────

class InferenceEngine:
    """通用推理引擎"""

    def __init__(self, config: InferConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._history: List[Dict[str, str]] = []
        self._device = None
        self._loaded = False
        self._stop_token_ids: list = []  # 收集所有停止 token ID（含 eos + eot）

    # ── 加载 ────────────────────────────────────────────────

    def load(self):
        """加载模型和 tokenizer"""
        cfg = self.config
        if self._verbose():
            print(f"[推理引擎] 加载基座模型: {cfg.base_path}")
            if cfg.adapter_path:
                print(f"[推理引擎] 加载 LoRA adapter: {cfg.adapter_path}")
            print(f"[推理引擎] 后端: {cfg.backend}, 4-bit: {cfg.use_4bit}")

        load_kwargs = dict(
            trust_remote_code=cfg.trust_remote_code,
        )

        # 4-bit 量化
        if cfg.use_4bit and cfg.backend in ("cuda", "rocm"):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = cfg.device_map
        else:
            if cfg.backend == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
            else:
                load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = cfg.device_map

        # ROCm attention 优化
        if cfg.backend == "rocm":
            attn = _detect_rocm_attn()
            if attn != "sdpa":
                load_kwargs["attn_implementation"] = attn

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_path, **load_kwargs
        )

        # Tokenizer: 优先从 adapter 加载
        tok_path = cfg.adapter_path or cfg.base_path
        tok_dir = Path(tok_path)
        if not (tok_dir / "tokenizer_config.json").exists():
            tok_path = cfg.base_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=cfg.trust_remote_code, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 收集所有停止 token：eos + eot (end of turn)
        # Gemma 等模型用 <turn|> 标记 turn 结束，仅靠 eos 无法正确停止
        # 优先从 generation_config 获取（可能包含多个 stop id），再从 tokenizer 补充
        self._stop_token_ids = []
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            gc_eos = getattr(self.model.generation_config, 'eos_token_id', None)
            if gc_eos is not None:
                if isinstance(gc_eos, list):
                    self._stop_token_ids.extend(gc_eos)
                elif isinstance(gc_eos, int):
                    self._stop_token_ids.append(gc_eos)
        # tokenizer 补充 (eos + eot)
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is not None:
                if isinstance(val, list):
                    self._stop_token_ids.extend(val)
                elif isinstance(val, int):
                    self._stop_token_ids.append(val)
        # 去重但保持顺序
        seen = set()
        self._stop_token_ids = [x for x in self._stop_token_ids if not (x in seen or seen.add(x))]

        # LoRA
        if cfg.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)

        self.model.eval()
        self.model.config.use_cache = True

        # BF16 fix
        if cfg.backend in ("cuda", "rocm"):
            try:
                base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
                if hasattr(base_model, "lm_head"):
                    base_model.lm_head = base_model.lm_head.to(torch.bfloat16)
            except Exception:
                pass

        # device
        if cfg.backend == "cuda" or cfg.backend == "rocm":
            self._device = torch.device("cuda")
        elif cfg.backend == "cpu":
            self._device = torch.device("cpu")

        self._loaded = True
        if self._verbose():
            self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        try:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if total > 1e9:
                size_str = f"{total/1e9:.1f}B"
            elif total > 1e6:
                size_str = f"{total/1e6:.1f}M"
            else:
                size_str = f"{total/1e3:.1f}K"
            gpu_name = ""
            if self.config.backend in ("cuda", "rocm"):
                gpu_name = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
            print(f"[推理引擎] 参数: {size_str}, 可训练: {trainable:,}{gpu_name}")
        except Exception:
            pass

    def _verbose(self) -> bool:
        return self.config.verbose

    def _move_to_device(self, inputs: dict) -> dict:
        """将 inputs 移动到正确的设备"""
        if self._device is not None:
            return {k: v.to(self._device) for k, v in inputs.items()}
        return inputs

    def _get_generate_kwargs(self, override: Optional[Dict[str, Any]] = None) -> dict:
        """构建 generate 参数"""
        cfg = self.config
        # 使用收集到的多停止 token（eos + eot），确保 Gemma 等模型正确停止
        stop_ids = self._stop_token_ids if self._stop_token_ids else self.tokenizer.eos_token_id
        kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )
        if override:
            kwargs.update(override)
        return kwargs

    # ── 单次推理 ────────────────────────────────────────────

    def generate(self, prompt: str, **override) -> str:
        """同步生成（非流式）

        参数:
            prompt: 用户输入文本
            **override: 覆盖生成参数 (max_new_tokens, temperature 等)

        返回:
            生成的回复字符串
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        messages = [{"role": "user", "content": prompt}]
        formatted = _build_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True)
        inputs = self._move_to_device(inputs)

        gen_kwargs = self._get_generate_kwargs(override)

        t0 = time.time()
        with torch.no_grad():
            if self.config.backend in ("cuda", "rocm"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        reply = _extract_new_tokens(self.tokenizer, outputs, inputs)

        if self._verbose():
            new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            tps = new_tokens / elapsed if elapsed > 0 else 0
            print(f"[推理引擎] {new_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s")

        return reply

    # ── 流式推理 ────────────────────────────────────────────

    def generate_stream(self, prompt: str, **override) -> Iterator[str]:
        """流式生成 — 逐 token 返回

        用法:
            for chunk in engine.generate_stream("你的问题"):
                print(chunk, end="", flush=True)
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        messages = [{"role": "user", "content": prompt}]
        formatted = _build_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True)
        inputs = self._move_to_device(inputs)

        gen_kwargs = self._get_generate_kwargs(override)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        t0 = time.time()
        thread = Thread(target=self._generate_thread, args=(inputs, gen_kwargs))
        thread.start()

        token_count = 0
        for text in streamer:
            token_count += 1
            yield text

        thread.join()
        elapsed = time.time() - t0
        if self._verbose():
            tps = token_count / elapsed if elapsed > 0 else 0
            print(f"\n[推理引擎] {token_count} tokens, {elapsed:.1f}s, {tps:.1f} tok/s")

    def _generate_thread(self, inputs, gen_kwargs):
        """在子线程中运行 generate"""
        with torch.no_grad():
            if self.config.backend in ("cuda", "rocm"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.model.generate(**inputs, **gen_kwargs)
            else:
                self.model.generate(**inputs, **gen_kwargs)

    # ── 多轮对话 ────────────────────────────────────────────

    def chat(self, user_input: str, **override) -> str:
        """多轮对话 — 自动管理历史

        用法:
            engine.chat("Hello")
            engine.chat("What's my name?")
            engine.clear_history()
        """
        self._history.append({"role": "user", "content": user_input})
        formatted = _build_chat_prompt(self.tokenizer, self._history)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True)
        inputs = self._move_to_device(inputs)

        gen_kwargs = self._get_generate_kwargs(override)

        t0 = time.time()
        with torch.no_grad():
            if self.config.backend in ("cuda", "rocm"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        reply = _extract_new_tokens(self.tokenizer, outputs, inputs)
        self._history.append({"role": "assistant", "content": reply})

        if self._verbose():
            new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            tps = new_tokens / elapsed if elapsed > 0 else 0
            print(f"[推理引擎] {new_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s")

        return reply

    def chat_stream(self, user_input: str, **override) -> Iterator[str]:
        """多轮流式对话"""
        self._history.append({"role": "user", "content": user_input})
        formatted = _build_chat_prompt(self.tokenizer, self._history)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True)
        inputs = self._move_to_device(inputs)

        gen_kwargs = self._get_generate_kwargs(override)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        full_reply = []
        thread = Thread(target=self._generate_thread, args=(inputs, gen_kwargs))
        thread.start()

        for text in streamer:
            full_reply.append(text)
            yield text

        thread.join()
        self._history.append({"role": "assistant", "content": "".join(full_reply)})

    def clear_history(self):
        """清除对话历史"""
        self._history = []

    def set_system_prompt(self, content: str):
        """设置系统提示词（会插入到历史开头）"""
        # 移除旧的系统消息
        self._history = [m for m in self._history if m.get("role") != "system"]
        self._history.insert(0, {"role": "system", "content": content})

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history.copy()

    # ── 批量推理 ────────────────────────────────────────────

    def generate_batch(
        self,
        prompts: List[str],
        **override,
    ) -> List[str]:
        """批量推理 — 逐个生成（适合小批量）

        参数:
            prompts: 问题列表
            **override: 生成参数覆盖

        返回:
            回复列表
        """
        results = []
        total = len(prompts)
        for i, p in enumerate(prompts):
            if self._verbose():
                print(f"[推理引擎] 批量推理 {i+1}/{total}")
            reply = self.generate(p, **override)
            results.append(reply)
        return results

    def generate_stream_batch(
        self,
        prompts: List[str],
        **override,
    ) -> Iterator[tuple]:
        """流式批量推理 — 逐条流式返回

        返回:
            Iterator of (index, prompt, reply_stream) 其中 reply_stream 是逐 token yield 的
        """
        for i, p in enumerate(prompts):
            if self._verbose():
                print(f"[推理引擎] 批量推理 {i+1}/{len(prompts)}")
            yield i, p, self.generate_stream(p, **override)

    # ── 状态 ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """释放模型显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._loaded = False
        if self._verbose():
            print("[推理引擎] 模型已卸载")


# ── 工具函数 ────────────────────────────────────────────────

def _detect_rocm_attn() -> str:
    """检测 ROCm 可用的最优 attention 实现"""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        pass
    try:
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"
    except Exception:
        pass
    return "eager"


def resolve_base_from_adapter(adapter_path: str) -> Optional[str]:
    """从 LoRA adapter_config.json 中读取基座模型路径"""
    adapter_cfg = Path(adapter_path) / "adapter_config.json"
    if not adapter_cfg.exists():
        return None
    try:
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            return None
        bp = Path(base_name)
        if bp.exists():
            return str(bp.resolve())
        # 在 models/ 下按名字匹配
        models_dir = BASE / "models"
        if models_dir.exists():
            for m in models_dir.iterdir():
                if m.is_dir() and (m.name == bp.name or m.name == base_name):
                    return str(m.resolve())
        return base_name  # HF ID
    except Exception:
        return None


def scan_for_models() -> List[Dict[str, Any]]:
    """扫描 models/ 目录返回可用基座模型列表"""
    from .scanner import scan_models
    return scan_models()


def scan_for_adapters() -> List[Dict[str, Any]]:
    """扫描 outputs/ 目录返回可用 LoRA adapter 列表"""
    from .scanner import scan_outputs
    return scan_outputs(details=True)
