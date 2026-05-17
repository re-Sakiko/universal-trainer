# Universal Trainer

通用大模型微调框架 — 支持 NVIDIA、AMD、Apple Silicon、Intel 及纯 CPU 训练。

## 特性

- **多后端**: CUDA / ROCm / DirectML / MPS / CPU 自动检测
- **4-bit QLoRA**: CUDA + ROCm 均支持 BF16 + 4-bit 量化
- **多格式**: ShareGPT, Alpaca, JSONL, CSV, Text 自动识别
- **格式校验**: 自动检测模型/数据集完整性，给出明确的缺失文件提示
- **LoRA 管理**: 推理时自动扫描 outputs/ 目录，支持交互式选择

## 快速开始

```bash
pip install -r requirements.txt

# 1. 把基座模型放到 models/ 目录
mkdir models
# 下载示例: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 2. 把数据集放到 datasets/ 目录
mkdir datasets
# 支持 .json / .jsonl / .csv / .txt

# 3. 扫描可用资源
python train.py --scan

# 4. 开始训练（交互式选择模型和数据）
python train.py

# 或直接指定路径
python train.py --model models/your-model --dataset datasets/your_data.json --max_steps 2000

# 5. 推理
python infer.py                               # 自动扫描 outputs/ 和 models/
python infer.py --model outputs/trained_model --test
python infer.py --model outputs/trained_model -q "你的问题"
```

## 命令参考

### train.py

```
python train.py
    交互式训练 — 自动扫描 models/ 和 datasets/，列出可用选项供选择

python train.py --scan
    扫描 models/ datasets/ outputs/ 目录，列出所有资源及校验状态

python train.py --validate
    详细校验 models/ 和 datasets/ 目录下所有文件的格式完整性

python train.py --model models/my-model --dataset datasets/my_data.json
    直接指定模型和数据集路径，跳过交互式选择

python train.py --model models/my-model --dataset data.csv --format csv
    手动指定数据格式（默认自动检测）
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动扫描 | 基座模型路径 |
| `--dataset` | 自动扫描 | 数据集路径 |
| `--format` | auto | 数据格式: auto / sharegpt / alpaca / jsonl / csv / text |
| `--max_steps` | 2000 | 训练步数 |
| `--max_samples` | 全部 | 最大样本数 |
| `--lr` | 2e-4 | 学习率 |
| `--batch_size` | 2 | 每设备批次大小 |
| `--grad_accum` | 4 | 梯度累积步数 |
| `--max_seq_length` | 1024 | 最大序列长度 |
| `--lora_r` | 16 | LoRA Rank |
| `--lora_alpha` | 32 | LoRA Alpha |
| `--backend` | auto | 后端: auto / cuda / rocm / directml / mps / cpu |
| `--output` | outputs/trained_model | 输出目录 |

### infer.py

```
python infer.py
    自动扫描 outputs/ 和 models/，交互式选择 LoRA 和基座模型

python infer.py --test
    使用默认路径进行测试推理

python infer.py --model outputs/my-lora --test
    测试指定 LoRA 适配器（自动检测基座模型）

python infer.py --model outputs/my-lora --base models/my-base-model
    手动指定基座模型路径

python infer.py --model outputs/my-lora -q "你的问题"
    单次问答

python infer.py --model outputs/my-lora
    交互式对话模式
```

## 支持的 GPU

| 后端 | 适用硬件 | 系统 | 精度 | 4-bit |
|------|----------|------|------|-------|
| CUDA | NVIDIA GPU | Win / Linux | BF16 | 是 |
| ROCm | AMD GPU | Linux | BF16 | 是 |
| DirectML | AMD / Intel / NVIDIA | Windows | FP16 | 否 |
| MPS | Apple M1-M4 | macOS | FP16 | 否 |
| CPU | 任何机器 | 全平台 | FP32 | 否 |

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

python train.py --dataset data.json --max_steps 2000
```

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
| Text | 纯文本，双换行分隔不同样本 |

格式自动检测，也可手动指定 `--format alpaca`。

## 目录结构

```
universal_trainer/
├── core/
│   ├── config.py          # 训练配置 + GPU/ROCm 自动检测
│   ├── engine.py          # 多后端训练引擎 (CUDA/DirectML/MPS/CPU)
│   ├── dataset_loader.py  # 多格式数据集加载器
│   ├── scanner.py         # 目录扫描器 (模型/数据集/LoRA)
│   └── validator.py       # 格式校验器
├── models/                # 基座模型目录 (支持递归扫描)
├── datasets/              # 训练数据目录
├── outputs/               # 训练输出 + LoRA 适配器目录
├── train.py               # 训练入口
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

# 交互式推理（自动扫描所有 LoRA）
python infer.py
```
