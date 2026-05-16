# Universal Trainer

通用大模型微调框架 — 支持所有主流 GPU 和数据集格式。

## 特性

- **多后端**: CUDA / DirectML / MPS / CPU 自动检测
- **多格式**: ShareGPT, Alpaca, JSONL, CSV, Text 自动识别
- **4-bit QLoRA**: NVIDIA GPU 上极致省显存
- **HTML 前端**: 浏览器配置参数、监控训练、在线推理
- **零依赖启动**: CPU 模式只需 `pip install torch transformers datasets peft trl`

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把基座模型放到 models/ 目录
#   例如: models/Qwen2-0.5B-Instruct/

# 3. CLI 训练
python train.py --dataset datasets/your_data.json --max_steps 2000

# 4. Web GUI (浏览器操作)
python train.py --web
# 打开 http://localhost:9999

# 5. 推理测试
python infer.py --model outputs/trained_model --test
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

## 支持的 GPU

| 后端 | 适用 | 精度 | 4-bit |
|------|------|------|-------|
| CUDA | NVIDIA GPU | BF16 | ✓ |
| DirectML | AMD / Intel / NVIDIA (Windows) | FP16 | ✗ |
| MPS | Apple M1-M4 | FP16 | ✗ |
| CPU | 任何机器 | FP32 | ✗ |

## 目录结构

```
universal_trainer/
├── core/              # 核心引擎
│   ├── config.py      # 训练配置
│   ├── dataset_loader.py  # 数据集加载
│   └── engine.py      # 训练引擎
├── web/               # Web 前端
│   ├── server.py      # Flask API
│   └── static/
│       └── index.html # GUI
├── models/            # 放基座模型
├── datasets/          # 放训练数据
├── outputs/           # 训练输出
├── train.py           # CLI 入口
└── infer.py           # 推理入口
```

## 高级用法

```bash
# 指定后端
python train.py --dataset data.json --backend cpu --max_steps 500

# 指定格式 + 限制样本
python train.py --dataset data.csv --format csv --max_samples 1000

# 自定义超参
python train.py --dataset data.json --lora_r 32 --lr 1e-4 --max_steps 3000
```
