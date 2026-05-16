# 训练数据集目录

把训练数据文件放到这里，程序会自动扫描并识别格式。

### 支持的格式

| 格式 | 文件 | 示例 |
|------|------|------|
| ShareGPT | `.json` | `[{"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}]` |
| Alpaca | `.json` | `[{"instruction":"...", "input":"...", "output":"..."}]` |
| JSONL | `.jsonl` | 每行一个 JSON 对象 |
| CSV | `.csv` | `question,answer` 列 |
| 纯文本 | `.txt` | 双换行分隔不同样本 |

### 使用方式

```bash
# 自动扫描 datasets/ 目录
python train.py --scan

# 或直接指定文件
python train.py --dataset datasets/my_data.json
```
