# 训练输出目录

训练完成的 LoRA 模型保存在这里。

### 输出结构

```
outputs/
└── my-model/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer_config.json
```

### 使用训练好的模型

```bash
python infer.py --model outputs/my-model --test
```
