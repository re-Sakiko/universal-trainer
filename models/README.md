# 基座模型目录

把 HuggingFace 格式的模型文件夹放到这里。支持递归扫描（子目录内也可检测）。

### 示例结构

```
models/
└── your-model-name/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### 推荐模型

| 模型 | 大小 | 下载 |
|------|------|------|
| Gemma-4-4B-it | ~15GB | `huggingface-cli download google/gemma-4-4B-it` |
| Llama-3.2-3B-Instruct | ~6GB | `huggingface-cli download meta-llama/Llama-3.2-3B-Instruct` |
| Qwen2.5-0.5B-Instruct | ~1GB | `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-1.5B-Instruct | ~3GB | `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct` |

### 自动检测

```bash
# 扫描 models/ 目录下所有模型并校验格式
python train.py --scan
python train.py --validate
```
