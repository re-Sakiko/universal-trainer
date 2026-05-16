# 基座模型目录

把 HuggingFace 格式的模型文件夹放到这里。

### 示例结构

```
models/
└── Qwen2-0.5B-Instruct/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### 推荐模型

| 模型 | 大小 | 下载 |
|------|------|------|
| Qwen2-0.5B-Instruct | ~1GB | `huggingface-cli download Qwen/Qwen2-0.5B-Instruct` |
| Qwen2.5-0.5B-Instruct | ~1GB | `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2-1.5B-Instruct | ~3GB | `huggingface-cli download Qwen/Qwen2-1.5B-Instruct` |

### 自动下载

```bash
# 程序会自动检测 models/ 目录下的模型
python train.py --scan
```
