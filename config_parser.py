#!/usr/bin/env python3
"""
YAML 配置解析器, 供 run_vllm_benchmark.sh 调用.

用法:
    # 全局配置
    python3 config_parser.py config.yaml global input_len
    python3 config_parser.py config.yaml global model_base

    # 模型列表 / 量化列表
    python3 config_parser.py config.yaml list-models
    python3 config_parser.py config.yaml list-all-quants
    python3 config_parser.py config.yaml list-quants qwen3-1.7b

    # 模型字段
    python3 config_parser.py config.yaml get qwen3-1.7b display
    python3 config_parser.py config.yaml get qwen3-1.7b tp
    python3 config_parser.py config.yaml get qwen3-1.7b gpu_mem_util

    # 模型路径 (自动展开 ${MODEL_BASE})
    python3 config_parser.py config.yaml get-path qwen3-1.7b bf16

    # 导出全局配置为 shell 变量 (供 eval 使用)
    python3 config_parser.py config.yaml export-globals

paths 中的 ${MODEL_BASE} 会被环境变量或 global.model_base 替换.
"""

import os
import re
import sys

import yaml


def expand_env(s: str, model_base: str = "") -> str:
    """展开字符串中的 ${VAR} 环境变量引用."""
    def replacer(m):
        var = m.group(1)
        # MODEL_BASE 优先用环境变量, fallback 到 YAML 的 global.model_base
        if var == "MODEL_BASE":
            return os.environ.get("MODEL_BASE", model_base)
        return os.environ.get(var, m.group(0))
    return re.sub(r'\$\{(\w+)\}', replacer, str(s))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <config.yaml> <command> [args...]",
              file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    command = sys.argv[2]
    args = sys.argv[3:]

    cfg = load_config(config_path)
    g = cfg.get("global", {})
    models = cfg.get("models", {})
    model_base = os.environ.get("MODEL_BASE", g.get("model_base", ""))

    if command == "global":
        # 获取一个全局配置值
        if not args:
            print("需要指定 key", file=sys.stderr)
            sys.exit(1)
        key = args[0]
        val = g.get(key, "")
        print(expand_env(str(val), model_base))

    elif command == "export-globals":
        # 导出全局配置为 KEY=VALUE 格式, 供 bash eval
        key_map = {
            "model_base":             "MODEL_BASE",
            "bench_mode":             "BENCH_MODE",
            "input_len":              "INPUT_LEN",
            "output_len":             "OUTPUT_LEN",
            "num_prompts":            "NUM_PROMPTS",
            "num_samples_per_prompt": "NUM_SAMPLES_PER_PROMPT",
            "max_concurrency":        "MAX_CONCURRENCY",
            "request_rate":           "REQUEST_RATE",
            "server_port":            "SERVER_PORT",
            "ascend_devices":         "ASCEND_RT_VISIBLE_DEVICES",
            "bucket_min":             "BUCKET_MIN",
            "bucket_max":             "BUCKET_MAX",
        }
        for yaml_key, bash_var in key_map.items():
            val = g.get(yaml_key, "")
            if val != "":
                print(f'{bash_var}="{val}"')

    elif command == "list-models":
        print(" ".join(models.keys()))

    elif command == "list-all-quants":
        all_quants = set()
        for m in models.values():
            all_quants.update(m.get("paths", {}).keys())
        print(" ".join(sorted(all_quants)))

    elif command == "list-quants":
        if not args:
            print("需要指定模型 key", file=sys.stderr)
            sys.exit(1)
        model_key = args[0]
        if model_key not in models:
            print(f"未知模型: {model_key}", file=sys.stderr)
            sys.exit(1)
        print(" ".join(models[model_key].get("paths", {}).keys()))

    elif command == "get":
        if len(args) < 2:
            print("需要: model_key field", file=sys.stderr)
            sys.exit(1)
        model_key, field = args[0], args[1]
        if model_key not in models:
            print(f"未知模型: {model_key}", file=sys.stderr)
            sys.exit(1)
        val = models[model_key].get(field, "")
        print(expand_env(str(val), model_base))

    elif command == "get-path":
        if len(args) < 2:
            print("需要: model_key quant", file=sys.stderr)
            sys.exit(1)
        model_key, quant = args[0], args[1]
        if model_key not in models:
            sys.exit(1)
        path = models[model_key].get("paths", {}).get(quant, "")
        if path:
            print(expand_env(str(path), model_base))

    else:
        print(f"未知命令: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
