#!/usr/bin/env python3
"""
模型 shape 配置模块

从 HuggingFace config.json 提取的精确架构参数，
用于计算 decode 阶段各线性层的 GEMM shape。

使用方式:
    from shapes import get_all_shapes, MODELS
"""

# ============================================================
# 模型架构参数 (来自 HuggingFace config.json)
# ============================================================

MODELS = {
    "Qwen3-30B-A3B": {
        "type": "moe",
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,       # GQA 8:1
        "head_dim": 128,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "intermediate_size": 6144,      # dense FFN (未使用, 全 MoE)
        "first_k_dense_replace": 0,
        "n_shared_experts": 0,
        "attention_type": "gqa",
        # MLA 相关 (不适用)
        "q_lora_rank": None,
        "kv_lora_rank": None,
    },
    "Pangu-718B": {
        "type": "moe",
        "hidden_size": 7680,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,     # MLA, 非 GQA
        "head_dim": 128,                # v_head_dim
        "moe_intermediate_size": 2048,
        "num_experts": 256,             # n_routed_experts
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "intermediate_size": 18432,     # dense FFN (前3层)
        "first_k_dense_replace": 3,
        "n_shared_experts": 1,
        "attention_type": "mla",
        # MLA 参数
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
}


def compute_expert_batch(total_batch: int, num_experts: int,
                         top_k: int) -> int:
    """计算每个 expert 平均分到的 token 数"""
    return max(1, total_batch * top_k // num_experts)


def get_moe_shapes(model_name: str, decode_batch: int) -> list[dict]:
    """
    获取 MoE 层的 GEMM shape 列表

    返回: [{"name": str, "M": int, "K": int, "N": int, "count_per_layer": int, "num_layers": int}, ...]
    """
    cfg = MODELS[model_name]
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    E = cfg["num_experts"]
    top_k = cfg["num_experts_per_tok"]
    num_moe_layers = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]

    B_e = compute_expert_batch(decode_batch, E, top_k)

    shapes = [
        {
            "name": "expert_gate_up",
            "M": B_e,
            "K": H,
            "N": I * 2,     # gate + up 融合
            "count_per_layer": top_k,  # 每个 token 激活 top_k 个 expert
            "num_layers": num_moe_layers,
        },
        {
            "name": "expert_down",
            "M": B_e,
            "K": I,
            "N": H,
            "count_per_layer": top_k,
            "num_layers": num_moe_layers,
        },
    ]

    # Shared expert (718B 独有, full batch)
    if cfg["n_shared_experts"] > 0:
        shapes.extend([
            {
                "name": "shared_expert_gate_up",
                "M": decode_batch,
                "K": H,
                "N": I * 2,
                "count_per_layer": 1,
                "num_layers": num_moe_layers,
            },
            {
                "name": "shared_expert_down",
                "M": decode_batch,
                "K": I,
                "N": H,
                "count_per_layer": 1,
                "num_layers": num_moe_layers,
            },
        ])

    return shapes


def get_attention_shapes(model_name: str, decode_batch: int) -> list[dict]:
    """
    获取 Attention 投影层的 GEMM shape 列表

    注意: Attention 投影走 Dense 路径 (npu_quant_matmul), 不是 MoE 路径
    """
    cfg = MODELS[model_name]
    H = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    B = decode_batch

    if cfg["attention_type"] == "gqa":
        n_heads = cfg["num_attention_heads"]
        n_kv_heads = cfg["num_key_value_heads"]
        head_dim = cfg["head_dim"]
        return [
            {"name": "q_proj", "M": B, "K": H, "N": n_heads * head_dim,
             "count_per_layer": 1, "num_layers": num_layers},
            {"name": "k_proj", "M": B, "K": H, "N": n_kv_heads * head_dim,
             "count_per_layer": 1, "num_layers": num_layers},
            {"name": "v_proj", "M": B, "K": H, "N": n_kv_heads * head_dim,
             "count_per_layer": 1, "num_layers": num_layers},
            {"name": "o_proj", "M": B, "K": n_heads * head_dim, "N": H,
             "count_per_layer": 1, "num_layers": num_layers},
        ]

    elif cfg["attention_type"] == "mla":
        q_lora = cfg["q_lora_rank"]
        kv_lora = cfg["kv_lora_rank"]
        n_heads = cfg["num_attention_heads"]
        qk_nope = cfg["qk_nope_head_dim"]
        qk_rope = cfg["qk_rope_head_dim"]
        v_head = cfg["v_head_dim"]
        return [
            # Q: hidden → q_lora_rank (压缩)
            {"name": "q_compress", "M": B, "K": H, "N": q_lora,
             "count_per_layer": 1, "num_layers": num_layers},
            # Q: q_lora_rank → heads * (nope + rope) (展开)
            {"name": "q_expand_nope", "M": B, "K": q_lora,
             "N": n_heads * qk_nope,
             "count_per_layer": 1, "num_layers": num_layers},
            {"name": "q_expand_rope", "M": B, "K": q_lora,
             "N": n_heads * qk_rope,
             "count_per_layer": 1, "num_layers": num_layers},
            # KV: hidden → kv_lora_rank (压缩, 这是 KV cache 存储的)
            {"name": "kv_compress", "M": B, "K": H, "N": kv_lora,
             "count_per_layer": 1, "num_layers": num_layers},
            # O: heads * v_head_dim → hidden
            {"name": "o_proj", "M": B, "K": n_heads * v_head,
             "N": H,
             "count_per_layer": 1, "num_layers": num_layers},
        ]

    return []


def get_dense_ffn_shapes(model_name: str, decode_batch: int) -> list[dict]:
    """获取 Dense FFN 层的 GEMM shape (仅 718B 前3层)"""
    cfg = MODELS[model_name]
    if cfg["first_k_dense_replace"] == 0:
        return []

    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    B = decode_batch
    n_dense = cfg["first_k_dense_replace"]

    return [
        {"name": "dense_gate_up", "M": B, "K": H, "N": I * 2,
         "count_per_layer": 1, "num_layers": n_dense},
        {"name": "dense_down", "M": B, "K": I, "N": H,
         "count_per_layer": 1, "num_layers": n_dense},
    ]


def get_all_shapes(model_name: str, decode_batch: int = 256) -> dict:
    """
    获取指定模型 decode 阶段所有线性层的 GEMM shape

    Args:
        model_name: 模型名 ("Qwen3-30B-A3B" 或 "Pangu-718B")
        decode_batch: decode 阶段并发序列数 (默认 256 = 32 prompts × n=8)

    Returns:
        {"attention": [...], "moe": [...], "dense_ffn": [...]}
    """
    return {
        "attention": get_attention_shapes(model_name, decode_batch),
        "moe": get_moe_shapes(model_name, decode_batch),
        "dense_ffn": get_dense_ffn_shapes(model_name, decode_batch),
    }


# ============================================================
# 直接运行: 打印所有 shape
# ============================================================

def _print_shapes(model_name: str, decode_batch: int = 256):
    """打印指定模型的所有 GEMM shape"""
    all_shapes = get_all_shapes(model_name, decode_batch)
    cfg = MODELS[model_name]
    B_e = compute_expert_batch(decode_batch, cfg["num_experts"],
                               cfg["num_experts_per_tok"])

    print(f"\n{'='*70}")
    print(f"  {model_name}  (decode_batch={decode_batch}, "
          f"tokens_per_expert≈{B_e})")
    print(f"{'='*70}")

    for category, shapes in all_shapes.items():
        if not shapes:
            continue
        print(f"\n  [{category}]")
        print(f"  {'Name':<25s} {'M':>5s} {'K':>6s} {'N':>6s}"
              f"  {'×/layer':>7s}  {'layers':>6s}")
        print(f"  {'-'*60}")
        for s in shapes:
            print(f"  {s['name']:<25s} {s['M']:>5d} {s['K']:>6d}"
                  f" {s['N']:>6d}  {s['count_per_layer']:>7d}"
                  f"  {s['num_layers']:>6d}")


if __name__ == "__main__":
    for model in MODELS:
        _print_shapes(model, decode_batch=256)
