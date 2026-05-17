# Universal Trainer

通用大模型微调框架 — 支持 NVIDIA、AMD、Apple Silicon、Intel 及纯 CPU 训练。

## 特性

- **多后端**: CUDA / ROCm / DirectML / MPS / CPU 自动检测
- **4-bit QLoRA**: CUDA + ROCm 均支持 BF16 + 4-bit 量化
- **多格式**: ShareGPT, Alpaca, JSONL, CSV, Text 自动识别
- **HTML 前端**: 浏览器配置参数、监控训练、在线推理
- **零依赖启动**: CPU 模式只需 `pip install torch transformers datasets peft trl`

## 快速开始

```bash
pip install -r requirements.txt

# 模型放到 models/ 目录，数据放到 datasets/ 目录
python train.py --dataset datasets/your_data.json --max_steps 2000

# Web GUI
python train.py --web              # 打开 http://localhost:9999

# 推理
python infer.py --model outputs/trained_model --test
```

## 支持的 GPU

| 后端 | 适用硬件 | 系统 | 精度 | 4-bit |
|------|----------|------|------|-------|
| CUDA | NVIDIA GPU | Win / Linux | BF16 | ✓ |
| ROCm | AMD GPU | Linux | BF16 | ✓ |
| DirectML | AMD / Intel / NVIDIA | Windows | FP16 | ✗ |
| MPS | Apple M1-M4 | macOS | FP16 | ✗ |
| CPU | 任何机器 | 全平台 | FP32 | ✗ |

### CUDA (NVIDIA)

```bash
pip install bitsandbytes>=0.45.0
python train.py --dataset data.json --max_steps 2000
```

### ROCm (AMD, Linux)

ROCm 与 CUDA 能力完全对等 — 支持 BF16、4-bit QLoRA、Flash Attention：

```bash
# 系统依赖: ROCm 5.7+ (https://rocm.docs.amd.com)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install bitsandbytes>=0.45.0

# 可选: Flash Attention 加速
pip install flash-attn --no-build-isolation

# 自动检测, 直接训练
python train.py --dataset data.json --max_steps 2000
```

框架自动应用 ROCm 优化：`PYTORCH_TUNABLEOP_ENABLED` 可调算子、Flash Attention 2、`torch.compile` inductor 编译、8-bit AdamW。

### DirectML (AMD / Intel, Windows)

```bash
pip install torch-directml>=0.2.5
python train.py --dataset data.json --backend directml
```

### MPS (Apple Silicon)

```bash
python train.py --dataset data.json --backend mps
```

### CPU

```bash
python train.py --dataset data.json --backend cpu --max_steps 500
```

## 数据集格式

| 格式 | 示例 |
|------|------|
| ShareGPT / OpenAI | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| Alpaca | `{"instruction": "...", "input": "...", "output": "..."}` |
| JSONL | 每行: `{"question": "...", "answer": "..."}` |
| CSV | `question,answer` (或其他列名组合) |
| Text | 纯文本, 双换行分隔不同样本 |

格式自动检测，也可手动指定 `--format alpaca`。

## 目录结构

```
universal_trainer/
├── core/                  # 核心引擎
│   ├── config.py          # 训练配置 + GPU 检测
│   ├── dataset_loader.py  # 多格式数据集加载
│   └── engine.py          # 多后端训练引擎
├── web/                   # Web GUI
│   ├── server.py          # Flask API
│   └── static/
│       └── index.html
├── models/                # 基座模型目录
├── datasets/              # 训练数据目录
├── outputs/               # 训练输出目录
├── train.py               # CLI 入口
└── infer.py               # 推理入口
```

## 高级用法

```bash
# 指定后端
python train.py --dataset data.json --backend rocm --max_steps 2000

# 指定格式 + 限制样本
python train.py --dataset data.csv --format csv --max_samples 1000

# 自定义超参
python train.py --dataset data.json --lora_r 32 --lr 1e-4 --max_steps 3000
```
