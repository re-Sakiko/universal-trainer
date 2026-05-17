"""
Web 前端服务器
==============
Flask REST API + 静态 HTML GUI

启动:
    python web/server.py
    python web/server.py --port 8080
"""

import sys, os, json, threading, time, argparse
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from flask import Flask, request, jsonify, send_from_directory
from core import TrainConfig, TrainingEngine, list_supported_formats, scan_models, scan_datasets, scan_outputs
from core.validator import validate_all_models, validate_all_datasets, validate_model_dir, validate_dataset_file

app = Flask(__name__, static_folder="static", static_url_path="")

# 全局训练引擎
_engine: TrainingEngine = None
_training_thread: threading.Thread = None


@app.route("/")
def index():
    static_dir = Path(__file__).parent / "static"
    return send_from_directory(str(static_dir), "index.html")


@app.route("/api/formats")
def api_formats():
    return jsonify(list_supported_formats())


@app.route("/api/scan")
def api_scan():
    return jsonify({
        "models": scan_models(str(BASE / "models")),
        "datasets": scan_datasets(str(BASE / "datasets")),
        "outputs": scan_outputs(str(BASE / "outputs")),
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    """创建训练配置"""
    global _engine
    data = request.json

    cfg = TrainConfig(
        model_name=data.get("model_name", "Qwen2-0.5B-Instruct"),
        model_path=data.get("model_path", "models/Qwen2-0.5B-Instruct"),
        dataset_path=data["dataset_path"],
        dataset_format=data.get("dataset_format", "auto"),
        max_samples=data.get("max_samples"),
        max_steps=data.get("max_steps", 2000),
        learning_rate=data.get("learning_rate", 2e-4),
        per_device_batch_size=data.get("batch_size", 2),
        gradient_accumulation_steps=data.get("grad_accum", 4),
        max_seq_length=data.get("max_seq_length", 1024),
        lora_r=data.get("lora_r", 16),
        lora_alpha=data.get("lora_alpha", 32),
        output_dir=data.get("output_dir", "outputs/trained_model"),
        backend=data.get("backend", "auto"),
    )

    _engine = TrainingEngine(cfg)
    return jsonify({"status": "ok", "backend": cfg.backend, "dtype": cfg.dtype})


@app.route("/api/prepare", methods=["POST"])
def api_prepare():
    """准备训练（加载模型+数据）"""
    global _engine
    if _engine is None:
        return jsonify({"error": "请先配置训练参数"}), 400

    def _run():
        try:
            _engine.prepare()
        except Exception as e:
            _engine._log(f"ERROR: {e}")

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    return jsonify({"status": "ok", "logs": _engine.get_logs()[-5:]})


@app.route("/api/train", methods=["POST"])
def api_train():
    """启动训练（后台线程）"""
    global _engine, _training_thread
    if _engine is None:
        return jsonify({"error": "请先准备训练"}), 400

    def _run():
        try:
            _engine.train()
        except Exception as e:
            _engine._log(f"TRAIN ERROR: {e}")

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _engine
    if _engine:
        _engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/logs")
def api_logs():
    """获取最新日志"""
    global _engine
    if _engine is None:
        return jsonify({"logs": []})

    after = request.args.get("after", type=int, default=0)
    logs = _engine.get_logs()
    new_logs = logs[after:] if after < len(logs) else []
    return jsonify({"logs": new_logs, "count": len(logs)})


@app.route("/api/status")
def api_status():
    global _engine, _training_thread
    training = _training_thread is not None and _training_thread.is_alive()
    return jsonify({
        "training": training,
        "prepared": _engine is not None and _engine.model is not None,
        "backend": _engine.config.backend if _engine else "none",
    })


@app.route("/api/validate")
def api_validate():
    """校验 models/ 和 datasets/ 格式"""
    return jsonify({
        "models": validate_all_models(str(BASE / "models")),
        "datasets": validate_all_datasets(str(BASE / "datasets")),
    })


@app.route("/api/validate/model/<path:model_name>")
def api_validate_model(model_name):
    """校验单个模型"""
    model_path = BASE / "models" / model_name
    if not model_path.exists():
        return jsonify({"error": "模型不存在"}), 404
    return jsonify(validate_model_dir(str(model_path)))


@app.route("/api/validate/dataset/<path:dataset_name>")
def api_validate_dataset(dataset_name):
    """校验单个数据集"""
    dataset_path = BASE / "datasets" / dataset_name
    if not dataset_path.exists():
        return jsonify({"error": "数据集不存在"}), 404
    return jsonify(validate_dataset_file(str(dataset_path)))


@app.route("/api/infer", methods=["POST"])
def api_infer():
    global _engine
    if _engine is None or _engine.model is None:
        return jsonify({"error": "模型未准备好"}), 400

    data = request.json
    question = data.get("question", "")
    max_tokens = data.get("max_tokens", 256)

    answer = _engine.infer(question, max_tokens)
    return jsonify({"answer": answer})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Universal Trainer Web UI")
    print(f"  http://{args.host}:{args.port}")
    print(f"  按 Ctrl+C 退出\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
