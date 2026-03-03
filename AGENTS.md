# AGENTS.md — verl-rollout-bench

## Project Overview

Benchmarking toolkit for comparing BF16 / W8A16 / W8A8 quantization on vLLM inference throughput,
targeting verl RL GRPO rollout scenarios on Ascend NPU. Four experiments in separate directories,
sharing a YAML config at the root.

## File Map

```
config.yaml                          # Model definitions, global params (shared by all experiments)
config_parser.py                     # YAML config parser, called by shell scripts (CLI tool)
001_multi_model_quant/               # Multi-model × multi-quant throughput comparison
  run_vllm_benchmark.sh              #   Main orchestrator: experiment matrix, server lifecycle
  summarize_benchmark.py             #   Result analysis: JSON → comparison tables (txt/csv/md)
002_decode_sweep/                    # Simulated GRPO rollout (mixed output_len)
  rollout_bench.py                   #   Core benchmark: vLLM LLM.generate() with per-request SamplingParams
  run_rollout_bench.sh               #   Shell orchestrator: BF16 vs W8A8D, NPU parallel, cleanup
  analyze_rollout.py                 #   Result comparison: throughput, speedup
003_operator_bench/                  # Single-operator benchmark (GEMM shapes)
  shapes.py                          #   Model GEMM shape definitions (30B-A3B, 718B)
  bench_ops.py                       #   BF16 vs W8A8D operator benchmark
  analyze.py                         #   Results analysis with validation
  run_operator_bench.sh              #   One-click: full benchmark + M-sweep + analysis
004_profiling/                       # Operator-level profiling
  profile_offline.sh                 #   torch_npu profiler + vllm bench throughput
  profile_online.sh                  #   msserviceprofiler + vllm serve
  analyze_profile.py                 #   Parse traces, compare BF16 vs W8A8D operator timing
```

## Build / Run / Test Commands

No formal build system, package manager, or test suite.

### Dependencies

- Python 3.10+ with `pyyaml`, `numpy`
- vLLM (with Ascend NPU support) — for benchmarking
- `torch`, `torch_npu` — for experiment 003 (operator benchmark)
- Standard library: `argparse`, `json`, `math`, `os`, `sys`, `re`, `pathlib`, `collections`, `gzip`

### Running

```bash
export MODEL_BASE="/path/to/models"

# Full benchmark (requires Ascend NPU + models)
bash 001_multi_model_quant/run_vllm_benchmark.sh

# Diagnostic mode (quick validation)
bash 001_multi_model_quant/run_vllm_benchmark.sh --diagnostic

# Subset of experiments
bash 001_multi_model_quant/run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16
```

### Config parser CLI

```bash
python3 config_parser.py config.yaml list-models
python3 config_parser.py config.yaml global input_len
python3 config_parser.py config.yaml get qwen3-1.7b tp
python3 config_parser.py config.yaml get-path qwen3-1.7b bf16
python3 config_parser.py config.yaml export-globals
```

### Linting (no configured linter)

```bash
# Suggested (not enforced):
ruff check --line-length 100
mypy --ignore-missing-imports config_parser.py
```

### Tests

No test suite exists. When adding tests, use `pytest`:

```bash
pytest tests/ -v
pytest tests/test_foo.py -v
pytest tests/test_foo.py::test_bar -v
```

## Code Style Guidelines

### Language

- **Comments and docstrings**: Chinese (中文). Intentional and consistent.
- **Variable/function names**: English, snake_case.
- **Commit messages**: English. `Fix:`, `Add:`, or descriptive imperative phrase. No conventional commits.
- **README files**: Chinese.

### Python Style

- **Shebang**: `#!/usr/bin/env python3` on all scripts.
- **Module docstring**: Chinese, describes purpose + usage examples.
- **Imports**: stdlib first, blank line, then third-party (`yaml`, `numpy`, `torch`).
  Side-effect imports use `# noqa: F401` (e.g., `import torch_npu  # noqa: F401`).
- **Type hints**: Python 3.10+ syntax: `tuple[X | None, Y]`, `list[int]`, `dict[str, dict]`.
  Never `Optional[X]` or `Tuple`.
- **String formatting**: f-strings exclusively. No `.format()` or `%`.
- **Naming**: snake_case for functions/variables. UPPER_CASE for module-level constants.
- **Keyword-only args**: Use `*` separator for clarity.
- **Error output**: `print(..., file=sys.stderr)` for errors, then `sys.exit(1)`. No exceptions for CLI.
- **No classes**: Module-level functions + `main()` pattern.
- **Line length**: ~100 chars (not strict).
- **Trailing commas**: Used in multi-line lists/dicts.
- **Docstrings**: Short Chinese one-liners for simple functions.
- **JSON output**: `json.dump(..., indent=2, ensure_ascii=False)`.
- **Timing**: `time.perf_counter()` for benchmarks, never `time.time()`.
- **CLI pattern**: `argparse` with `add_mutually_exclusive_group` for mode selection.
  All analysis scripts support `--markdown` flag for Markdown output.
- **Path handling**: Mix of `pathlib.Path` and `os.path`. New code should prefer `pathlib`.

### Bash Style

- **Header**: `set -euo pipefail`.
- **Functions**: snake_case. Inline comments in Chinese.
- **Variables**: UPPER_CASE for globals, `local` + lower_case for locals.
- **Quoting**: Always `"${VAR}"`.
- **Logging**: Color-coded: `log_info`, `log_ok`, `log_warn`, `log_error`, `log_step`.
- **Process management**: `kill_process_tree()` for recursive cleanup. Trap EXIT.
- **Config access**: Through `cfg()` wrapper calling `config_parser.py`.
- **Command building**: String concatenation with `+=`, then `eval`.

### YAML Config (config.yaml)

- `global:` (shared params) + `models:` (per-model config).
- Model fields: `display`, `tp`, `gpu_mem_util`, `paths` (keyed by quant).
- `${MODEL_BASE}` in paths, expanded at runtime.
- Adding a model: new block under `models:`. Missing quant paths auto-skipped.

### Error Handling

- **Python**: `print(f"错误信息", file=sys.stderr)` then `sys.exit(1)`.
- **Bash**: `return 1` from functions. Failed experiments logged but don't abort the full run.

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

1. **Offline mode preferred**: `LLM.generate()` directly, not HTTP. `n=8` multi-sampling
   simulates verl rollout's actual workload.
2. **Per-model gpu_memory_utilization**: Simulates verl's shared GPU memory (actor + rollout).
3. **Config-driven**: All model/experiment parameters in `config.yaml`. Shell scripts are generic.
4. **Process tree cleanup**: vLLM spawns worker subprocesses. `kill_process_tree()` ensures full
   cleanup between experiments to prevent GPU memory leaks.

### Adding New Experiments

1. Add model entry in `config.yaml` under `models:`.
2. No code changes needed — shell scripts auto-discover models from YAML.
3. Missing quant paths are auto-skipped (no error).
