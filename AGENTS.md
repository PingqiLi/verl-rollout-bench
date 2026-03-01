# AGENTS.md — verl-rollout-bench

## Project Overview

Benchmarking toolkit for comparing BF16 / W8A16 / W8A8 quantization on vLLM inference throughput,
targeting verl RL GRPO rollout scenarios on Ascend NPU. Organized as 4 experiments in separate directories,
sharing a YAML config at the root.

## File Map

```
config.yaml                          # Model definitions, global params (shared by all experiments)
config_parser.py                     # YAML config parser, called by shell scripts (CLI tool)
001_multi_model_quant/               # Experiment 001: multi-model × multi-quant throughput comparison
  run_vllm_benchmark.sh              #   Main orchestrator: experiment matrix, server lifecycle, profiling
  summarize_benchmark.py             #   Result analysis: JSON → comparison tables (txt/csv/markdown)
  result.txt                         #   Sample benchmark results
  rollout_bench_tables.tex           #   LaTeX output table
002_decode_sweep/                    # Experiment 002: decode length sweep (BF16 vs W8A8D)
  run_sweep.sh                       #   Sweep output_len 256..16384 with input_len=1 (pure decode)
  analyze_sweep.py                   #   Speedup vs output_len analysis
003_operator_bench/                  # Experiment 003: single-operator benchmark (GEMM shapes)
  shapes.py                          #   Model GEMM shape definitions (30B-A3B, 718B)
  bench_ops.py                       #   BF16 vs W8A8D operator benchmark
  analyze.py                         #   Results analysis with validation
004_profiling/                       # Experiment 004: operator-level profiling
  profile_offline.sh                 #   torch_npu profiler + vllm bench throughput
  profile_online.sh                  #   msserviceprofiler + vllm serve
  analyze_profile.py                 #   Parse traces, compare BF16 vs W8A8D operator timing
```

## Build / Run / Test Commands

There is no formal build system, package manager, or test suite.

### Dependencies

- Python 3.10+ with `pyyaml` (`import yaml`)
- vLLM (with Ascend NPU support) — for actual benchmarking
- Standard library: `argparse`, `json`, `math`, `os`, `sys`, `re`, `pathlib`, `collections`

### Running

```bash
# Full benchmark (requires Ascend NPU + models)
export MODEL_BASE="/path/to/models"
bash 001_multi_model_quant/run_vllm_benchmark.sh

# Diagnostic mode (quick validation)
bash 001_multi_model_quant/run_vllm_benchmark.sh --diagnostic

# Subset of experiments
bash 001_multi_model_quant/run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# Offline mode (default, recommended for throughput comparison)
bash 001_multi_model_quant/run_vllm_benchmark.sh --offline

# Online mode (latency metrics)
bash 001_multi_model_quant/run_vllm_benchmark.sh --online

### Summarize results independently

```bash
python3 001_multi_model_quant/summarize_benchmark.py outputs/<timestamp>/results --markdown
```

### Config parser CLI (used by shell script, can also run standalone)

```bash
python3 config_parser.py config.yaml list-models
python3 config_parser.py config.yaml global input_len
python3 config_parser.py config.yaml get qwen3-1.7b tp
python3 config_parser.py config.yaml get-path qwen3-1.7b bf16
python3 config_parser.py config.yaml export-globals
```

### Linting (no configured linter)

No flake8, ruff, mypy, or pre-commit configured. If adding linting, respect current style:

```bash
# Suggested (not enforced):
ruff check --line-length 100
mypy --ignore-missing-imports config_parser.py summarize_benchmark.py
```

### Tests

No test suite exists. When adding tests, use `pytest`:

```bash
pytest tests/ -v              # Run all
pytest tests/test_foo.py -v   # Single file
pytest tests/test_foo.py::test_bar -v  # Single test
```

## Code Style Guidelines

### Language

- **Comments and docstrings**: Chinese (中文). This is intentional and consistent across the project.
- **Variable/function names**: English, snake_case.
- **Commit messages**: English. Short imperative style: `Fix: description` or `Add: description`.
- **README**: Chinese.

### Python Style

- **Shebang**: `#!/usr/bin/env python3` on all scripts.
- **Module docstring**: Chinese, describes purpose + usage examples.
- **Imports**: stdlib first, then third-party (`yaml`). No blank line between stdlib groups.
  ```python
  import argparse
  import json
  import os
  import sys
  from collections import OrderedDict
  from pathlib import Path
  ```
- **Type hints**: Used for function signatures. Use `tuple[X | None, Y | None]` style (Python 3.10+),
  not `Optional[X]` or `Tuple`.
  ```python
  def _infer_output_tput(d: dict, *, default_input_len: int) -> tuple[float | None, str | None]:
  ```
- **String formatting**: f-strings exclusively. No `.format()` or `%`.
- **Naming**: snake_case for functions/variables. UPPER_CASE for module-level constants.
  ```python
  METRICS = [...]
  SPEEDUP_METRICS = [...]
  def normalize_result(data: dict, *, default_input_len: int) -> dict:
  ```
- **Keyword-only args**: Use `*` separator for clarity in functions with multiple params.
- **Error output**: `print(..., file=sys.stderr)` for errors, then `sys.exit(1)`.
- **No classes**: Scripts use module-level functions + `main()` pattern. Keep it simple.
- **Line length**: ~100 chars (not strict, no formatter configured).
- **Trailing commas**: Used in multi-line lists/dicts.
- **Docstrings**: Short Chinese one-liners for simple functions. Multi-line for complex ones.

### Bash Style

- **Header**: `set -euo pipefail` — strict error handling.
- **Functions**: snake_case. Documented with inline comments (Chinese).
- **Variables**: UPPER_CASE for globals/config, lower_case for locals.
  ```bash
  RESULT_DIR="${RUN_DIR}/results"   # Global
  local model_key="$1"              # Local
  ```
- **Quoting**: Always double-quote variable expansions: `"${VAR}"`.
- **Logging**: Color-coded functions: `log_info`, `log_ok`, `log_warn`, `log_error`, `log_step`.
- **Process management**: `kill_process_tree()` for recursive cleanup. Trap EXIT for cleanup.
- **Config access**: Through `cfg()` wrapper calling `config_parser.py`.
- **Command building**: String concatenation with `+=` for complex commands, then `eval`.

### YAML Config (config.yaml)

- **Structure**: `global:` (shared params) + `models:` (per-model config).
- **Model block fields**: `display`, `tp`, `gpu_mem_util`, `paths` (keyed by quant).
- **Variable expansion**: `${MODEL_BASE}` in paths, expanded at runtime.
- **Adding a model**: Add a new block under `models:` with display/tp/gpu_mem_util/paths.
  Missing quant paths = that combination is automatically skipped.

### Error Handling

- **Python**: `sys.exit(1)` with Chinese error message to stderr. No exceptions for CLI errors.
  ```python
  print(f"未知模型: {model_key}", file=sys.stderr)
  sys.exit(1)
  ```
- **Bash**: `return 1` from functions, checked by callers. Failed experiments logged but don't abort
  the full run (graceful degradation).

### Git Conventions

- **Commit prefix**: `Fix:`, `Add:`, or descriptive imperative phrase.
- **Examples from history**:
  ```
  Fix: NO_PROFILE unbound variable (missing initialization)
  YAML config + simplified output structure
  Default: offline mode, no profiling (matches verl behavior)
  ```
- **No conventional commits** (no `feat:`, `chore:` etc.).

## Architecture Notes

### Data Flow

```
config.yaml → config_parser.py → run_vllm_benchmark.sh → vLLM CLI
                                                        ↓
                                              outputs/<ts>/results/*.json
                                                        ↓
                                           summarize_benchmark.py → summary.{txt,csv,md}
```

### Key Design Decisions

1. **Offline mode preferred**: `LLM.generate()` directly, not HTTP. Supports `n=8` multi-sampling
   to simulate verl rollout's actual workload.
2. **Per-model gpu_memory_utilization**: Simulates verl's shared GPU memory (actor + rollout).
   Override with `--gpu-mem-util 0.9` for peak performance testing.
3. **Config-driven**: All model/experiment parameters in `config.yaml`. Shell script is generic.
4. **Process tree cleanup**: vLLM spawns worker subprocesses. `kill_process_tree()` ensures full
   cleanup between experiments to prevent GPU memory leaks.

### Adding New Experiments

1. Add model entry in `config.yaml` under `models:`.
2. No code changes needed — the shell script auto-discovers models from YAML.
3. Missing quant paths are auto-skipped (no error).
