"""Native dense Qwen3 decoder-only model (eager PyTorch reference attention).

M3 runs one sequence at a time: ``input_ids`` and ``positions`` are 1-D ``[N]`` tensors
for a single prompt, and attention is plain causal attention without a KV cache.
Batched/paged attention is connected to the scheduler's block tables in M4.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nano_vllm.config import get_rope_theta
from nano_vllm.layers.activation import SiluAndMul
from nano_vllm.layers.attention import Attention
from nano_vllm.layers.layernorm import RMSNorm
from nano_vllm.layers.linear import MergedLinear, QKVLinear
from nano_vllm.layers.rotary_embedding import RotaryEmbedding


class Qwen3Attention(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // self.num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVLinear(
            config.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads, bias
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, get_rope_theta(config)
        )
        self.attn = Attention(
            self.num_heads, self.num_kv_heads, self.head_dim, self.scaling, layer_idx
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q.view(n, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim))
        v = v.view(n, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)  # [N, num_heads, head_dim]
        return self.o_proj(o.reshape(n, self.q_size))

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Eager causal attention for one sequence; GQA repeats each KV head."""
        group = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            is_causal=True, scale=self.scaling,
        )
        return o.transpose(0, 1)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(config.hidden_size, [config.intermediate_size] * 2)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            positions, self.input_layernorm(hidden_states)
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    # checkpoint module name -> (packed module name, shard id); used by utils.loader.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return final hidden states ``[N, hidden_size]``."""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return logits ``[N, vocab_size]``."""
        return self.lm_head(hidden_states)
