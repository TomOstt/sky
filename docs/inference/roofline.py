#!/usr/bin/env python3
"""Roofline model for batch-1 LLM inference on Apple silicon.

Every number in docs/inference/part04-hardware-limits.md comes from this
file, so the arithmetic can be checked and re-run:

    python3 roofline.py              # all tables, Markdown
    python3 roofline.py --model qwen3-32b --hw m5-max-40

No dependencies. Model shapes come from each model's config.json on
Hugging Face. Hardware numbers carry a `source` string; anything marked
ESTIMATE must be replaced by a measurement before it is quoted.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

GB = 1e9


# ---------------------------------------------------------------- models

@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int  # dense MLP width (per expert for MoE)
    vocab: int
    tied: bool  # lm_head shares the embedding matrix
    experts: int = 0  # 0 = dense
    experts_active: int = 0
    source: str = ""

    # Parameter counts. Norm weights are ignored: < 0.01% of any model here.
    def attn_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        return q + kv + o

    def mlp_params(self) -> int:  # one expert (gate, up, down)
        return 3 * self.hidden * self.ffn

    def router_params(self) -> int:
        return self.hidden * self.experts if self.experts else 0

    def layer_params(self) -> int:
        n = max(self.experts, 1)
        return self.attn_params() + n * self.mlp_params() + self.router_params()

    def active_layer_params(self) -> int:
        n = self.experts_active if self.experts else 1
        return self.attn_params() + n * self.mlp_params() + self.router_params()

    def embed_params(self) -> int:
        return self.vocab * self.hidden

    def total_params(self) -> int:
        head = 0 if self.tied else self.embed_params()
        return self.layers * self.layer_params() + self.embed_params() + head

    def kv_bytes_per_position(self, bytes_per_elem: float) -> float:
        """K and V for one context position, all layers."""
        return self.layers * 2 * self.kv_heads * self.head_dim * bytes_per_elem


MODELS = {
    m.name: m
    for m in [
        Model("qwen3-0.6b", 1024, 28, 16, 8, 128, 3072, 151936, True,
              source="huggingface.co/Qwen/Qwen3-0.6B/config.json"),
        Model("qwen3-1.7b", 2048, 28, 16, 8, 128, 6144, 151936, True,
              source="huggingface.co/Qwen/Qwen3-1.7B/config.json"),
        Model("qwen3-4b", 2560, 36, 32, 8, 128, 9728, 151936, True,
              source="huggingface.co/Qwen/Qwen3-4B/config.json"),
        Model("qwen3-8b", 4096, 36, 32, 8, 128, 12288, 151936, False,
              source="huggingface.co/Qwen/Qwen3-8B/config.json"),
        Model("qwen3-14b", 5120, 40, 40, 8, 128, 17408, 151936, False,
              source="huggingface.co/Qwen/Qwen3-14B/config.json"),
        Model("qwen3-32b", 5120, 64, 64, 8, 128, 25600, 151936, False,
              source="huggingface.co/Qwen/Qwen3-32B/config.json"),
        Model("qwen3-30b-a3b", 2048, 48, 32, 4, 128, 768, 151936, False,
              experts=128, experts_active=8,
              source="huggingface.co/Qwen/Qwen3-30B-A3B/config.json"),
        Model("llama-3.1-8b", 4096, 32, 32, 8, 128, 14336, 128256, False,
              source="huggingface.co/meta-llama/Llama-3.1-8B/config.json"),
        Model("llama-3.1-70b", 8192, 80, 64, 8, 128, 28672, 128256, False,
              source="huggingface.co/meta-llama/Llama-3.1-70B/config.json"),
    ]
}


# ---------------------------------------------------------------- formats

@dataclass(frozen=True)
class Quant:
    name: str
    bits_per_weight: float  # including scales/biases/zero points
    source: str


QUANTS = {
    q.name: q
    for q in [
        # MLX affine: 4-bit values + one fp16 scale + one fp16 bias per group.
        Quant("mlx-q4-g64", 4 + 32 / 64, "mlx affine, group 64: 4 + 2*16/64"),
        Quant("mlx-q4-g32", 4 + 32 / 32, "mlx affine, group 32"),
        Quant("mlx-q3-g64", 3 + 32 / 64, "mlx affine, group 64"),
        Quant("mlx-q6-g64", 6 + 32 / 64, "mlx affine, group 64"),
        Quant("mlx-q8-g64", 8 + 32 / 64, "mlx affine, group 64"),
        # GGUF: Q4_0 = 32 weights in 18 bytes; Q4_K = 256 in 144; Q6_K = 256 in 210;
        # Q8_0 = 32 in 34. Q4_K_M mixes Q4_K with Q6_K for some tensors (~4.85 avg).
        Quant("gguf-q4_0", 18 * 8 / 32, "ggml block_q4_0: 2 + 16 bytes / 32"),
        Quant("gguf-q4_k", 144 * 8 / 256, "ggml block_q4_K: 144 bytes / 256"),
        Quant("gguf-q6_k", 210 * 8 / 256, "ggml block_q6_K: 210 bytes / 256"),
        Quant("gguf-q8_0", 34 * 8 / 32, "ggml block_q8_0: 34 bytes / 32"),
        Quant("mxfp4", 4 + 8 / 32, "OCP MX: e2m1 + one e8m0 scale per 32"),
        Quant("bf16", 16, "bfloat16"),
    ]
}


# ---------------------------------------------------------------- hardware

@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_gbs: float  # spec unified-memory bandwidth
    achievable_frac: float  # best streaming kernel / spec (measured where possible)
    fp16_tflops: float  # dense matmul peak usable by prefill (GPU ALUs or NAs)
    int8_tops: float  # 0 if no faster int8 path
    memory_gb: int
    source: dict = field(default_factory=dict)
    alu_tflops: float = 0.0  # shader-ALU fp16/fp32 peak (small-M verify runs here)

    def achievable_gbs(self) -> float:
        return self.bandwidth_gbs * self.achievable_frac

    def ridge_flop_per_byte(self) -> float:
        return self.fp16_tflops * 1e12 / (self.achievable_gbs() * GB)


# From research/R6-apple-metal.md (hardware model) and R3-mlx-core.md.
# "achievable" for the M2 Pro is the measured raw-Metal read kernel (196 GB/s,
# R6); MLX's own streaming kernels reach 182-184 GB/s (R3). For M5-family
# chips achievable bandwidth is unmeasured: 0.90 is the planning number
# (R6: plan for 520-585 GB/s on M5 Max). Neural-accelerator rates are LIKELY
# (two independent secondary measurements), not Apple-published.
HARDWARE = {
    h.name: h
    for h in [
        Hardware("m2-pro-16", 200, 0.98, 5.73, 0, 16,
                 {"bandwidth": "Apple spec 200 GB/s",
                  "achievable": "VERIFIED: raw Metal read 196 GB/s (R6); MLX 182-184 (R3)",
                  "fp16": "VERIFIED: 16 x 128 x 2 x 1.398 GHz = 5.73; MPP matmul2d reaches 5.42 (R6)"},
                 alu_tflops=5.73),
        Hardware("m4-max-40", 546, 0.90, 16.2, 0, 128,
                 {"bandwidth": "Apple spec", "achievable": "ESTIMATE (planning 90%)",
                  "fp16": "ESTIMATE: 40 x 128 x 2 x ~1.58 GHz"},
                 alu_tflops=16.2),
        Hardware("m5-pro-20", 307, 0.90, 32.8, 61.0, 64,
                 {"bandwidth": "VERIFIED: Apple spec 307 GB/s (R6)",
                  "achievable": "ESTIMATE (planning 90%)",
                  "fp16": "LIKELY: half of M5 Max NA rate (20 of 40 cores)",
                  "int8": "LIKELY/EXPERIMENTAL: half of M5 Max"},
                 alu_tflops=8.3),
        Hardware("m5-max-32", 460, 0.90, 52.0, 98.0, 36,
                 {"bandwidth": "VERIFIED: Apple spec 460 GB/s, 36 GB only (R6)",
                  "achievable": "ESTIMATE (planning 90%)",
                  "fp16": "LIKELY: 32/40 of the 40-core NA rate"},
                 alu_tflops=13.3),
        Hardware("m5-max-40", 614, 0.87, 65.0, 122.0, 128,
                 {"bandwidth": "VERIFIED: Apple spec 614 GB/s (R6)",
                  "achievable": "LIKELY: 0.87 = a contributor's measured ~530 GB/s (R3/R4); R6 plans 520-585; measure on day 1",
                  "fp16": "LIKELY: neural accelerators ~64-65.5 TFLOPS (R6, two sources)",
                  "int8": "LIKELY/EXPERIMENTAL: ~122 TOPS (R6, one source)"},
                 alu_tflops=16.6),
        Hardware("m5-ultra-80", 1200, 0.90, 130.0, 244.0, 512,
                 {"bandwidth": "VERIFIED: Apple spec 1.2 TB/s, shipped 2026-09-22 (R6)",
                  "achievable": "ESTIMATE (planning 90%)",
                  "fp16": "LIKELY: 2x M5 Max"},
                 alu_tflops=33.2),
    ]
}


# ---------------------------------------------------------------- decode

def decode_weight_bytes(m: Model, q: Quant, head_q: Quant | None = None) -> float:
    """Bytes of weights streamed per decode token (batch 1).

    All layer weights (active experts only, for MoE) plus the output head.
    The embedding table is gathered (one row) and not counted.
    """
    head_q = head_q or q
    layers = m.layers * m.active_layer_params() * q.bits_per_weight / 8
    head = m.embed_params() * head_q.bits_per_weight / 8
    return layers + head


def decode_bytes(m: Model, q: Quant, ctx: int, kv_bytes: float = 2.0) -> float:
    return decode_weight_bytes(m, q) + ctx * m.kv_bytes_per_position(kv_bytes)


def decode_ceiling_tps(m: Model, q: Quant, hw: Hardware, ctx: int,
                       kv_bytes: float = 2.0, spec: bool = False) -> float:
    """Tokens/s if decode streamed exactly its bytes at full speed.

    spec=True uses the datasheet bandwidth; otherwise the achievable fraction.
    """
    bw = hw.bandwidth_gbs if spec else hw.achievable_gbs()
    return bw * GB / decode_bytes(m, q, ctx, kv_bytes)


# ---------------------------------------------------------------- prefill

def prefill_flops_per_token(m: Model, prompt: int) -> float:
    """Average FLOPs per prompt token: 2 per weight (active) + causal attention.

    The output head runs once per prompt (last token only), so it is spread
    over the prompt. Attention: QK^T and PV, 2 FLOPs per MAC, causal half.
    """
    linear = 2 * m.layers * m.active_layer_params()
    head = 2 * m.embed_params() / prompt
    attn = 2 * 2 * m.layers * m.heads * m.head_dim * (prompt + 1) / 2
    return linear + head + attn


def prefill_ceiling_tps(m: Model, hw: Hardware, prompt: int, eff: float = 1.0,
                        int8: bool = False) -> float:
    peak = (hw.int8_tops if int8 and hw.int8_tops else hw.fp16_tflops) * 1e12
    return peak * eff / prefill_flops_per_token(m, prompt)


def ttft_s(m: Model, q: Quant, hw: Hardware, prompt: int, prefill_eff: float,
           decode_eff_frac: float, int8: bool = False) -> float:
    """Warm-process time to first token: prefill the prompt, then one decode step.

    prefill_eff: fraction of matmul peak achieved; decode_eff_frac: fraction of
    the achievable-bandwidth ceiling achieved.
    """
    t_prefill = prompt / prefill_ceiling_tps(m, hw, prompt, prefill_eff, int8)
    t_decode = 1 / (decode_ceiling_tps(m, q, hw, prompt) * decode_eff_frac)
    return t_prefill + t_decode


# ---------------------------------------------------------------- speculation

def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """Leviathan et al. 2023, eq. 1: tokens per target pass with gamma drafts
    and i.i.d. per-token acceptance alpha (includes the bonus token)."""
    if alpha >= 1:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speculative_tps(base_tps: float, alpha: float, gamma: int,
                    draft_cost: float, verify_overhead: float) -> float:
    """Effective tokens/s.

    base_tps: plain decode speed of the target model.
    draft_cost: time of one draft token / time of one target decode step.
    verify_overhead: extra time of a (gamma+1)-token verify pass relative to a
    1-token step (0.10 = 10% slower). Bandwidth-bound verification is close to 0
    while gamma+1 stays under the ridge point (see verify_free_tokens).
    """
    round_time = gamma * draft_cost + (1 + verify_overhead)
    return base_tps * expected_tokens_per_round(alpha, gamma) / round_time


def verify_free_tokens(m: Model, q: Quant, hw: Hardware, alu: bool = False) -> float:
    """How many tokens one weight stream can verify before compute dominates.

    Per token in the batch each weight costs 2 FLOPs; the weight bytes are read
    once. The pass stays bandwidth-bound while
        n * 2 * params <= ridge * weight_bytes.
    alu=True uses the shader-ALU peak (small-M kernels run there: neither MLX
    nor llama.cpp routes M below ~8-33 to the neural accelerators, R6).
    """
    params = m.layers * m.active_layer_params() + m.embed_params()
    peak = (hw.alu_tflops if alu and hw.alu_tflops else hw.fp16_tflops) * 1e12
    ridge = peak / (hw.achievable_gbs() * GB)
    return ridge * decode_weight_bytes(m, q) / (2 * params)


# ---------------------------------------------------------------- memory

def memory_gb(m: Model, q: Quant, ctx: int, kv_bytes: float = 2.0,
              scratch_gb: float = 0.3, runtime_gb: float = 0.2) -> dict:
    head = 0 if m.tied else m.embed_params()
    weights = (m.layers * m.layer_params() + m.embed_params() + head) * q.bits_per_weight / 8
    kv = ctx * m.kv_bytes_per_position(kv_bytes)
    return {
        "weights": weights / GB,
        "kv": kv / GB,
        "scratch": scratch_gb,
        "runtime": runtime_gb,
        "total": weights / GB + kv / GB + scratch_gb + runtime_gb,
    }


# ---------------------------------------------------------------- report

def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def report(model: str, hw_name: str) -> str:
    m, hw = MODELS[model], HARDWARE[hw_name]
    q = QUANTS["mlx-q4-g64"]
    s = [f"## {model} on {hw_name}\n"]
    s.append(f"Parameters: {m.total_params() / 1e9:.2f} B total, "
             f"{(m.layers * m.active_layer_params() + m.embed_params()) / 1e9:.2f} B touched per decode token "
             f"(source: {m.source}).\n")
    s.append(f"Hardware: {hw.bandwidth_gbs:.0f} GB/s spec, {hw.achievable_gbs():.0f} GB/s achievable "
             f"(assumed {hw.achievable_frac:.0%}); {hw.fp16_tflops:.1f} TFLOPS fp16 matmul; "
             f"ridge {hw.ridge_flop_per_byte():.0f} FLOP/byte. Sources: {hw.source}\n")

    rows = []
    for qn in ["mlx-q4-g64", "gguf-q4_k", "mlx-q6-g64", "mlx-q8-g64", "bf16"]:
        qq = QUANTS[qn]
        for ctx in [512, 4096, 32768]:
            rows.append([qn, ctx, f"{decode_bytes(m, qq, ctx) / GB:.2f}",
                         f"{decode_ceiling_tps(m, qq, hw, ctx, spec=True):.1f}",
                         f"{decode_ceiling_tps(m, qq, hw, ctx):.1f}"])
    s.append("### Decode ceiling (batch 1, fp16 KV)\n")
    s.append(table(rows, ["weights", "context", "GB/token", "tok/s @ spec BW",
                          "tok/s @ achievable BW"]) + "\n")

    rows = []
    for p in [128, 512, 1024, 4096, 8192, 32768]:
        rows.append([p, f"{prefill_flops_per_token(m, p) / 1e9:.1f}",
                     f"{prefill_ceiling_tps(m, hw, p):.0f}",
                     f"{prefill_ceiling_tps(m, hw, p, 0.7):.0f}",
                     f"{prefill_ceiling_tps(m, hw, p, 0.7, int8=True):.0f}" if hw.int8_tops else "n/a",
                     f"{ttft_s(m, q, hw, p, 0.7, 0.85) * 1000:.0f}"])
    s.append("### Prefill ceiling and warm TTFT\n")
    s.append(table(rows, ["prompt", "GFLOP/token", "tok/s @100% peak", "tok/s @70%",
                          "tok/s @70% int8", "TTFT ms (70% / 85%)"]) + "\n")

    rows = []
    for ctx in [4096, 32768, 131072]:
        for kvb, kvn in [(2.0, "fp16"), (1.0, "int8"), (0.5 + 4 / 64, "q4-g64")]:
            mem = memory_gb(m, q, ctx, kvb)
            rows.append([ctx, kvn, f"{mem['weights']:.2f}", f"{mem['kv']:.2f}", f"{mem['total']:.2f}"])
    s.append("### Memory (weights mlx-q4-g64; scratch 0.3 GB and runtime 0.2 GB assumed)\n")
    s.append(table(rows, ["context", "KV", "weights GB", "KV GB", "total GB"]) + "\n")

    base = decode_ceiling_tps(m, q, hw, 1024) * 0.9
    rows = []
    for alpha in [0.6, 0.7, 0.8]:
        for gamma in [3, 5, 7]:
            for dc in [0.02, 0.05]:
                rows.append([alpha, gamma, dc, f"{expected_tokens_per_round(alpha, gamma):.2f}",
                             f"{speculative_tps(base, alpha, gamma, dc, 0.10):.1f}"])
    s.append(f"### Speculative decoding (base {base:.1f} tok/s = 90% of the 1K-context ceiling; "
             f"verify overhead 10%)\n")
    s.append(f"Tokens one weight stream can verify before compute dominates: "
             f"{verify_free_tokens(m, q, hw):.1f} on the matrix peak, "
             f"{verify_free_tokens(m, q, hw, alu=True):.1f} on shader ALUs\n")
    s.append(table(rows, ["alpha", "gamma", "draft cost", "tokens/round", "effective tok/s"]) + "\n")
    return "\n".join(s)


# ---------------------------------------------------------------- per-kernel model
#
# Part XXII (docs/inference/part22-performance-model.md). One decode or prefill
# step is a list of kernels; each kernel costs its roofline time plus a fill
# term, and each kernel boundary costs a bubble:
#
#   T = sum_k [ max(bytes_k / (eta_k * BW_stream), lane_ops_k / R_lane,
#                   na_flops_k / (phi * F_na)) + t_fill ]
#       + N_dep * t_dep + N_indep * t_indep + exposed CPU time
#
# `python3 roofline.py --part22` prints every table of Part XXII and the
# per-kernel tables of Part IX. The default output above is unchanged.

@dataclass(frozen=True)
class GpuTiming:
    stream_gbs: float      # DRAM rate a well-written 4-bit GEMV streams at
    t_dep_us: float        # bubble at a dependent kernel boundary (tiny kernels)
    t_indep_us: float      # cost of an independent dispatch (concurrent encoder)
    t_fill_us: float       # per-kernel ramp-up, tail drain and latency floor
    lane_ops_per_s: float  # shader-ALU lane instructions per second (FMA = 1)
    roundtrip_us: float    # GPU idle gap if the CPU waits before committing
    source: dict = field(default_factory=dict)


GPU_TIMING = {
    "m2-pro-16": GpuTiming(
        185.0, 1.2, 0.14, 2.3, 16 * 128 * 1.398e9, 105.0,
        {"stream": "VERIFIED: raw-Metal 4-bit GEMV 183.8-186.3 GB/s on 32B shapes (R6 B2)",
         "t_dep, t_indep": "VERIFIED: 1.14-1.25 / 0.14 us (R6 B3)",
         "t_fill": "EXPERIMENTAL: 2.3 us from llama.cpp's 64 us/layer over ~20 dispatches and "
                   "~15 barriers (R5 B, D2); replace with the G0 chain microbenchmark",
         "lanes": "VERIFIED: 16 cores x 128 lanes x 1.398 GHz (R3 7.D)",
         "roundtrip": "VERIFIED: 104.6 us idle gap (R6 B3)"}),
    "m5-max-40": GpuTiming(
        534.0, 1.0, 0.14, 2.3, 40 * 128 * 1.62e9, 105.0,
        {"stream": "LIKELY: planning value = roofline 'achievable' (contributor 530 GB/s, R3-issues)",
         "t_dep": "EXPERIMENTAL: R6 assumes ~1 us on M5; measure",
         "t_fill": "EXPERIMENTAL: M2 Pro value carried over; measure at G5",
         "lanes": "LIKELY: 40 x 128 x ~1.62 GHz (R6, R8 4.3)",
         "roundtrip": "EXPERIMENTAL: M2 Pro value"}),
}

C_DEQ = 2.0  # ALU lane-ops per weight to unpack 4-bit affine (amortized), R3 3.3 / R8 4.3


@dataclass(frozen=True)
class Kern:
    name: str
    dram_bytes: float          # DRAM bytes per instance (weights + activations + KV)
    flops: float               # arithmetic per instance (reported)
    lane_ops: float = 0.0      # ALU lane instructions per instance
    na_flops: float = 0.0      # FLOPs issued to the neural accelerators / MMA path
    eta: float = 1.0           # fraction of stream_gbs this kernel class reaches
    dependent: bool = True     # preceded by a dependent boundary (barrier)
    count: int = 1             # instances per step (64 = once per layer)
    weight_bytes: float = 0.0  # the weight part of dram_bytes (reported)


def kern_time_us(k: Kern, t: GpuTiming, hw: Hardware, phi: float = 0.8,
                 overlap: bool = True) -> tuple:
    """Time of one instance (without its boundary) and the bound that sets it."""
    mem = k.dram_bytes / (k.eta * t.stream_gbs * GB) * 1e6
    alu = k.lane_ops / t.lane_ops_per_s * 1e6
    mma = 0.0
    if k.na_flops:
        if hw.int8_tops:  # chip with neural accelerators: matrix work runs there
            mma = k.na_flops / (phi * hw.fp16_tflops * 1e12) * 1e6
        else:             # simdgroup MMA shares the ALUs with the dequantization
            alu += k.na_flops / 2 / (phi * t.lane_ops_per_s) * 1e6
    if not overlap:
        mma, alu = mma + alu, 0.0
    best = max((mem, "DRAM"), (alu, "ALU"), (mma, "matrix"))
    return best[0] + t.t_fill_us, best[1]


def step_us(ks: list, t: GpuTiming, hw: Hardware, n_dep: int | None = None,
            n_indep: int | None = None, phi: float = 0.8, overlap: bool = True) -> dict:
    work = sum(k.count * kern_time_us(k, t, hw, phi, overlap)[0] for k in ks)
    n = sum(k.count for k in ks)
    dep = n_dep if n_dep is not None else sum(k.count for k in ks if k.dependent)
    indep = n_indep if n_indep is not None else n - dep
    bubbles = dep * t.t_dep_us + indep * t.t_indep_us
    return {"work_us": work, "bubbles_us": bubbles, "total_us": work + bubbles,
            "dispatches": n, "dependent": dep, "fill_us": n * t.t_fill_us}


def sky_decode_kernels(m: Model, q: Quant, ctx: int, M: int = 1, kv_bytes: float = 2.0,
                       b_kv: int = 256, attn_eta: float = 0.9, na: bool = False,
                       rows_per_sg: int = 4, stats_kernel: bool = False) -> list:
    """Sky's fused plan (Part IX section 3): 5 kernels per layer (+1 combine when
    the context has more than one split) and 2 head kernels. M = verify width.
    stats_kernel=False recomputes the RMSNorm statistics in every simdgroup of the
    consuming GEMV (Part IX section 4); True stores them from a small kernel."""
    bpw = q.bits_per_weight / 8
    H, hd, nq, nkv, F, L, V = (m.hidden, m.head_dim, m.heads, m.kv_heads, m.ffn,
                                m.layers, m.vocab)
    act, res = 2, 4  # fp16 activations, fp32 residual stream
    w_qkv, w_o, w_gu, w_d = H * (nq + 2 * nkv) * hd, nq * hd * H, 2 * H * F, F * H
    S = max(1, math.ceil(ctx / b_kv))
    partial = (hd + 2) * 4 * nq * M

    def gemv(w: float) -> dict:
        if na and M > 1:  # FMAs on the matrix units, dequant on the ALU
            return {"lane_ops": C_DEQ * w, "na_flops": 2 * M * w}
        return {"lane_ops": (C_DEQ + M) * w}

    ks = [Kern("S1/S2 RMSNorm statistics (stored)", M * (H * 4 + 64), M * 2 * H,
               count=2 * L)] if stats_kernel else []
    ks += [
        Kern("K1 rmsnorm + QKV GEMV + q/k-norm + RoPE + KV append",
             w_qkv * bpw + M * (H * res + nq * hd * act + 2 * nkv * hd * kv_bytes) + H * 2 + 1024,
             2 * M * w_qkv + M * (3 * H + 6 * (nq + nkv) * hd), count=L,
             weight_bytes=w_qkv * bpw,
             **{k: v + (k == "lane_ops") * (not stats_kernel) * M * ((nq + 2 * nkv) * hd / rows_per_sg) * H
                for k, v in gemv(w_qkv).items()}),
        Kern("K2 GQA attention, fixed-size splits",
             ctx * nkv * hd * 2 * kv_bytes + M * nq * hd * act + (S > 1) * S * partial,
             M * (4 * nq * hd * ctx + 5 * nq * ctx), eta=attn_eta, count=L),
    ]
    if S > 1:
        ks.append(Kern("K2c deterministic split combine", S * partial + M * nq * hd * act,
                       M * 6 * S * nq * hd, count=L))
    ks += [
        Kern("K3 o_proj GEMV + residual", w_o * bpw + M * (nq * hd * act + 2 * H * res),
             2 * M * w_o + M * H, count=L, weight_bytes=w_o * bpw, **gemv(w_o)),
        Kern("K4 rmsnorm + gate/up GEMV + SwiGLU",
             w_gu * bpw + M * (H * res + F * act) + H * 2,
             2 * M * w_gu + M * (3 * H + 4 * F), count=L, weight_bytes=w_gu * bpw,
             **{k: v + (k == "lane_ops") * (not stats_kernel) * M * (2 * F / rows_per_sg) * H
                for k, v in gemv(w_gu).items()}),
        Kern("K5 down GEMV + residual", w_d * bpw + M * (F * act + 2 * H * res),
             2 * M * w_d + M * H, count=L, weight_bytes=w_d * bpw, **gemv(w_d)),
        Kern("H1 final norm + lm_head GEMV + partial argmax", V * H * bpw + M * H * res,
             2 * M * V * H, count=1, weight_bytes=V * H * bpw, **gemv(V * H)),
        Kern("H2 sample combine: token, position, stop flag", M * 64 * 1024, M * 4 * V / 64,
             count=1),
    ]
    return ks


def mlx_like_decode_kernels(m: Model, q: Quant, ctx: int, kv_bytes: float = 2.0,
                            attn_eta: float = 0.52) -> list:
    """mlx-lm's Qwen3 decode step (R4 3.1): 19 dispatches per layer + 9 per token.
    Only the GEMVs and attention move real bytes; the rest are latency-bound."""
    bpw = q.bits_per_weight / 8
    H, hd, nq, nkv, F, L, V = (m.hidden, m.head_dim, m.heads, m.kv_heads, m.ffn,
                                m.layers, m.vocab)
    small = lambda name, n: Kern(name, 4 * H * 2, 4 * H, count=n * L)  # noqa: E731
    gemv = lambda name, w, n=1: Kern(name, w * bpw, 2 * w, lane_ops=(C_DEQ + 1) * w,  # noqa: E731
                                     count=n * L, weight_bytes=w * bpw)
    return [
        small("rms_norm x2 (input, post-attention)", 2),
        gemv("q_proj", H * nq * hd), gemv("k_proj, v_proj", H * nkv * hd, 2),
        small("q_norm, k_norm", 2), small("rope q, rope k", 2), small("KV writes k, v", 2),
        Kern("sdpa_vector", ctx * nkv * hd * 2 * kv_bytes, 4 * nq * hd * ctx,
             eta=attn_eta, count=L),
        gemv("o_proj", nq * hd * H), small("residual adds", 2),
        gemv("gate_proj, up_proj", H * F, 2), small("SwiGLU (compiled)", 1),
        gemv("down_proj", F * H),
        Kern("per token: embed gather x3 + dequant, final norm, logsumexp, subtract, argmax",
             8 * H * 2, 8 * H, count=8),
        Kern("lm_head", V * H * bpw, 2 * V * H, lane_ops=(C_DEQ + 1) * V * H, count=1,
             weight_bytes=V * H * bpw),
    ]


def sky_prefill_kernels(m: Model, q: Quant, T: int, ctx0: int, kv_bytes: float = 2.0,
                        BM: int = 64, BN: int = 128) -> list:
    """One prefill chunk of T tokens after ctx0 cached tokens (Part IX section 7)."""
    bpw = q.bits_per_weight / 8
    H, hd, nq, nkv, F, L = m.hidden, m.head_dim, m.heads, m.kv_heads, m.ffn, m.layers
    act = 2
    tiles_m = math.ceil(T / BM)
    gemm = lambda name, w, n_out, extra: Kern(  # noqa: E731
        name, w * bpw + T * extra, 2 * T * w, lane_ops=C_DEQ * w * tiles_m,
        na_flops=2 * T * w, count=L, weight_bytes=w * bpw)
    qkv_rows = (nq + 2 * nkv) * hd
    attn_flops = 4 * nq * hd * T * (ctx0 + (T + 1) / 2)
    return [
        gemm("P1 QKV GEMM (+ norm, q/k-norm, RoPE, KV append)", H * qkv_rows, qkv_rows,
             H * act + qkv_rows * act),
        Kern("P2 causal flash attention", (ctx0 + T) * nkv * hd * 2 * kv_bytes + 2 * T * nq * hd * act,
             attn_flops, na_flops=attn_flops, count=L),
        gemm("P3 o_proj GEMM + residual", nq * hd * H, H, nq * hd * act + 2 * H * 4),
        gemm("P4 gate/up GEMM + SwiGLU (+ norm)", 2 * H * F, 2 * F, H * act + F * act),
        gemm("P5 down GEMM + residual", F * H, H, F * act + 2 * H * 4),
    ]


def fmt_us(x: float) -> str:
    return f"{x / 1000:.2f} ms" if x >= 1000 else f"{x:.1f} us"


def part22_report() -> str:
    q = QUANTS["mlx-q4-g64"]
    s = ["# Part XXII tables (generated by roofline.py --part22)\n"]

    # -- Part IX: per-kernel bytes and FLOPs, one layer, decode M = 1
    for model in ["qwen3-32b", "qwen3-8b"]:
        m = MODELS[model]
        rows = []
        for ctx in [512, 32768]:
            for k in sky_decode_kernels(m, q, ctx):
                if k.count != m.layers:
                    continue
                if ctx == 32768 and not k.name.startswith("K2"):
                    continue
                rows.append([f"{k.name} (ctx {ctx})" if k.name.startswith("K2") else k.name,
                             f"{k.weight_bytes / 1e6:.2f}", f"{(k.dram_bytes - k.weight_bytes) / 1e6:.3f}",
                             f"{k.flops / 1e6:.1f}", f"{k.flops / k.dram_bytes:.2f}"])
        s.append(f"## Part IX: Sky decode kernels per layer, {model} (MB, MFLOP)\n")
        s.append(table(rows, ["kernel", "weight MB", "other DRAM MB", "MFLOP",
                              "FLOP/byte"]) + "\n")

    # -- Validation against mlx-lm on the M2 Pro (R4)
    hw, t = HARDWARE["m2-pro-16"], GPU_TIMING["m2-pro-16"]
    cases = [  # model, KV MB in the timed window, dispatches, dependent, CPU ms, measured ms
        ("qwen3-8b", 23.3, 693, 480, 2.2, 0.52, [25.79, 26.42]),
        ("qwen3-0.6b", 18.6, 541, 373, 1.94, 1.0, [3.55, 3.76]),
    ]
    rows = []
    for name, kv_mb, n, dep, cpu_ms, eta, meas in cases:
        m = MODELS[name]
        ks = mlx_like_decode_kernels(m, q, 0, attn_eta=eta)
        r = step_us(ks, t, hw, n_dep=dep, n_indep=n - dep)
        attn = kv_mb * 1e6 / (eta * t.stream_gbs * GB) * 1e6
        gpu = r["total_us"] + attn
        exposed = max(0.0, cpu_ms * 1000 - gpu)
        pred = gpu + exposed
        for mm in meas:
            rows.append([name, f"{decode_weight_bytes(m, q) / 1e6:.1f} + {kv_mb:.1f}", n, dep,
                         fmt_us(r["work_us"] - r["fill_us"] + attn), fmt_us(r["fill_us"]),
                         fmt_us(r["bubbles_us"]), fmt_us(exposed), fmt_us(pred),
                         f"{1e6 / pred:.1f}", f"{mm:.2f} ms ({1000 / mm:.1f})",
                         f"{(pred / 1000 / mm - 1) * 100:+.1f}%"])
    s.append("## Validation: mlx-lm decode on the M2 Pro, predicted vs measured (R4)\n")
    s.append(table(rows, ["model", "weights + KV MB", "dispatches", "dependent",
                          "streaming", "fill", "bubbles", "CPU exposed", "predicted",
                          "pred tok/s", "measured (tok/s)", "error"]) + "\n")

    # -- Worked example: Qwen3-32B decode on M5 Max, Sky vs MLX-like
    m, hw = MODELS["qwen3-32b"], HARDWARE["m5-max-40"]
    tA = GPU_TIMING["m5-max-40"]
    for ctx in [512]:
        rows = []
        for k in sky_decode_kernels(m, q, ctx):
            tk, bound = kern_time_us(k, tA, hw)
            rows.append([k.name, k.count, f"{k.dram_bytes / 1e6:.2f}", f"{k.flops / 1e6:.1f}",
                         bound, fmt_us(tk), fmt_us(k.count * (tk + (tA.t_dep_us if k.dependent else tA.t_indep_us)))])
        r = step_us(sky_decode_kernels(m, q, ctx), tA, hw)
        rows.append(["total", r["dispatches"], "", "", "", "", fmt_us(r["total_us"])])
        s.append(f"## Worked example: Sky plan, qwen3-32b on m5-max-40, context {ctx} "
                 f"(parameter set A)\n")
        s.append(table(rows, ["kernel", "per token", "MB each", "MFLOP each", "bound",
                              "time each", "time per token incl. boundary"]) + "\n")

    def sets():
        A = GPU_TIMING["m5-max-40"]
        yield "A planning (534 GB/s, t_fill 2.3 us)", A
        # B: stream rate at which the MLX-like plan reproduces the published 28.4 tok/s
        target = 1e6 / 28.4
        lo, hi = 300.0, 614.0
        for _ in range(60):
            mid = (lo + hi) / 2
            tm = GpuTiming(mid, A.t_dep_us, A.t_indep_us, A.t_fill_us, A.lane_ops_per_s,
                           A.roundtrip_us)
            ks = mlx_like_decode_kernels(m, q, 64)
            n = sum(k.count for k in ks)
            tot = step_us(ks, tm, hw, n_dep=round(n * 480 / 693))["total_us"]
            lo, hi = (mid, hi) if tot > target else (lo, mid)
        yield f"B fitted to MLX 28.4 tok/s ({lo:.0f} GB/s, t_fill 2.3 us)", GpuTiming(
            lo, A.t_dep_us, A.t_indep_us, A.t_fill_us, A.lane_ops_per_s, A.roundtrip_us)
        yield "C low overhead (534 GB/s, t_fill 0.5 us)", GpuTiming(
            534.0, A.t_dep_us, A.t_indep_us, 0.5, A.lane_ops_per_s, A.roundtrip_us)

    rows = []
    for label, tm in sets():
        for ctx in [512, 8192, 32768]:
            sky = step_us(sky_decode_kernels(m, q, ctx), tm, hw)
            ks = mlx_like_decode_kernels(m, q, ctx)
            n = sum(k.count for k in ks)
            mlx = step_us(ks, tm, hw, n_dep=round(n * 480 / 693))
            ceil_tps = tm.stream_gbs * GB / decode_bytes(m, q, ctx)
            rows.append([label, ctx, f"{ceil_tps:.1f}", sky["dispatches"],
                         fmt_us(sky["total_us"]), f"{1e6 / sky['total_us']:.1f}",
                         n, fmt_us(mlx["total_us"]), f"{1e6 / mlx['total_us']:.1f}",
                         f"{mlx['total_us'] / sky['total_us']:.3f}"])
    s.append("## Worked example: qwen3-32b decode on m5-max-40, Sky plan vs MLX-like plan\n")
    s.append("MLX-like attention efficiency 0.52 (R4, 8B at 4K on M2 Pro); Sky's 0.90 is a "
             "target. Ceiling = stream rate / bytes per token.\n")
    s.append(table(rows, ["parameter set", "context", "ceiling tok/s", "Sky dispatches",
                          "Sky step", "Sky tok/s", "MLX-like dispatches", "MLX-like step",
                          "MLX-like tok/s", "Sky / MLX-like"]) + "\n")

    # -- Small and draft models: Sky plan vs MLX-like plan (GPU time only)
    rows = []
    for hwn in ["m2-pro-16", "m5-max-40"]:
        hh, tt = HARDWARE[hwn], GPU_TIMING[hwn]
        for name, eta in [("qwen3-0.6b", 1.0), ("qwen3-1.7b", 1.0), ("qwen3-8b", 0.52)]:
            mm = MODELS[name]
            sky = step_us(sky_decode_kernels(mm, q, 256), tt, hh)["total_us"]
            ks = mlx_like_decode_kernels(mm, q, 256, attn_eta=eta)
            n = sum(k.count for k in ks)
            mlx = step_us(ks, tt, hh, n_dep=round(n * 480 / 693))["total_us"]
            rows.append([hwn, name, f"{1e6 / sky:.0f}", f"{1e6 / mlx:.0f}", f"{mlx / sky:.2f}",
                         f"{decode_ceiling_tps(mm, q, hh, 256):.0f}"])
    s.append("## Small and draft models, context 256: predicted GPU-side decode (tok/s); "
             "mlx-lm's CPU time (1.94 ms/token for 0.6B on M2 Pro, R4) can bound it further\n")
    s.append(table(rows, ["chip", "model", "Sky plan", "MLX-like plan", "ratio",
                          "roofline ceiling"]) + "\n")

    # -- Prefill model
    rows = []
    for name, hwn, prompts in [("qwen3-32b", "m5-max-40", [128, 512, 2048, 8192]),
                               ("qwen3-8b", "m2-pro-16", [512, 2048])]:
        mm, hh, tt = MODELS[name], HARDWARE[hwn], GPU_TIMING[hwn]
        for P in prompts:
            for phi in ([0.71, 0.85] if hh.int8_tops else [0.85, 0.95]):
                for overlap in [True, False]:
                    chunk = min(P, 512)
                    total, ctx0 = 0.0, 0
                    while ctx0 < P:
                        T = min(chunk, P - ctx0)
                        ks = sky_prefill_kernels(mm, q, T, ctx0)
                        total += step_us(ks, tt, hh, phi=phi, overlap=overlap)["total_us"]
                        ctx0 += T
                    head = step_us(sky_decode_kernels(mm, q, P)[-2:], tt, hh)["total_us"]
                    total += head
                    rows.append([name, hwn, P, phi, "yes" if overlap else "no",
                                 f"{P / (total / 1e6):.0f}",
                                 f"{prefill_ceiling_tps(mm, hh, P):.0f}"])
    s.append("## Prefill model (chunk 512; phi = fraction of matrix peak; overlap = "
             "dequantization overlaps the matrix work)\n")
    s.append(table(rows, ["model", "chip", "prompt", "phi", "overlap", "predicted tok/s",
                          "ceiling @100% peak"]) + "\n")

    # -- Verify cost and speculation
    rows = []
    t1 = {}
    for hwn in ["m2-pro-16", "m5-max-40"]:
        hh, tt = HARDWARE[hwn], GPU_TIMING[hwn]
        for model in ["qwen3-8b", "qwen3-32b"]:
            mm = MODELS[model]
            base = step_us(sky_decode_kernels(mm, q, 1024), tt, hh)["total_us"]
            t1[(hwn, model)] = base
            for Mv in [2, 4, 8, 16, 32]:
                rec = step_us(sky_decode_kernels(mm, q, 1024, M=Mv), tt, hh)["total_us"]
                sto = step_us(sky_decode_kernels(mm, q, 1024, M=Mv, stats_kernel=True), tt, hh)["total_us"]
                alu = min(rec, sto)
                cell = [hwn, model, Mv, f"{rec / base:.2f}", f"{sto / base:.2f}",
                        f"{alu / base:.2f}", f"{(alu / base - 1) / (Mv - 1):.3f}"]
                if hh.int8_tops:
                    na = min(step_us(sky_decode_kernels(mm, q, 1024, M=Mv, na=True, stats_kernel=sk),
                                     tt, hh)["total_us"] for sk in (False, True))
                    cell += [f"{na / base:.2f}", f"{(na / base - 1) / (Mv - 1):.3f}"]
                else:
                    cell += ["-", "-"]
                rows.append(cell)
    s.append("## Verify cost predicted by the model, context 1024 (ratio to M = 1; "
             "beta = marginal cost per extra row)\n")
    s.append(table(rows, ["chip", "model", "M", "ALU, stats recomputed", "ALU, stats stored",
                          "ALU best", "ALU beta", "NA best", "NA beta"]) + "\n")

    measured = {  # verify-cost ratios T(M)/T(1), whole forward
        "mlx M5 Max 70B (mlx#3791)": {1: 1.0, 3: 84.8 / 75.9, 5: 112.7 / 75.9, 8: 170.8 / 75.9},
        "llama.cpp M2 Pro 8B Q4_0 (R5 G)": {1: 1.0, 2: 1.52, 3: 1.66, 4: 2.08, 5: 2.43,
                                            8: 3.76},
        "mlx-lm M2 Pro 8B (R4 7.D)": {1: 1.0, 2: 1.73, 3: 2.52, 4: 3.33, 5: 4.10, 8: 6.48},
    }

    def ratio_at(curve: dict, M: int) -> float:
        xs = sorted(curve)
        for a, b in zip(xs, xs[1:]):
            if a <= M <= b:
                return curve[a] + (curve[b] - curve[a]) * (M - a) / (b - a)
        return curve[xs[-1]]

    m, hw = MODELS["qwen3-32b"], HARDWARE["m5-max-40"]
    tA = GPU_TIMING["m5-max-40"]
    base = t1[("m5-max-40", "qwen3-32b")]
    draft = step_us(sky_decode_kernels(MODELS["qwen3-0.6b"], q, 1024), tA, hw)["total_us"]
    c_draft = draft / base
    rows = []
    for tau, gamma, label in [(2.29, 3, "EAGLE-3 HumanEval"), (2.49, 3, "EAGLE-3 GSM8K"),
                              (3.08, 7, "EAGLE-3 GSM8K"), (4.4, 8, "DFlash-class (none public)")]:
        M = gamma + 1
        cells = [label, tau, gamma]
        model_ratio = min(step_us(sky_decode_kernels(m, q, 1024, M=M, na=M > 8, stats_kernel=sk),
                                  tA, hw)["total_us"] for sk in (False, True)) / base
        betas = [(model_ratio - 1) / gamma, 0.05] + [
            (ratio_at(c, M) - 1) / gamma for c in measured.values()]
        for beta in betas:
            S = tau / (1 + gamma * 0.03 + gamma * beta)
            cells.append(f"{beta:.3f}: {S:.2f}x ({1e6 / base * S:.0f})")
        rows.append(cells)
    s.append(f"## Speculation on qwen3-32b / m5-max-40: S = tau / (1 + gamma c + gamma beta), "
             f"c = 0.03, base {1e6 / base:.1f} tok/s (Sky plan, context 1024); cells are "
             f"'beta: speedup (tok/s)'. A 0.6B draft costs c = {c_draft:.3f} in Sky's plan.\n")
    s.append(table(rows, ["drafter", "tau", "gamma", "Sky model", "Sky target",
                          *measured.keys()]) + "\n")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--hw", default=None)
    ap.add_argument("--part22", action="store_true",
                    help="print the per-kernel tables of Parts IX and XXII")
    a = ap.parse_args()
    if a.part22:
        print(part22_report())
        return
    pairs = [(a.model, a.hw)] if a.model and a.hw else [
        ("qwen3-32b", "m5-max-40"), ("qwen3-32b", "m5-ultra-80"),
        ("qwen3-32b", "m4-max-40"), ("qwen3-8b", "m2-pro-16"),
        ("qwen3-0.6b", "m2-pro-16")]
    print("# Roofline tables (generated by roofline.py)\n")
    rows = []
    for name, m in MODELS.items():
        rows.append([name, f"{m.total_params() / 1e9:.2f}",
                     f"{decode_weight_bytes(m, QUANTS['mlx-q4-g64']) / GB:.2f}",
                     f"{m.kv_bytes_per_position(2.0) / 1024:.0f}"])
    print(table(rows, ["model", "params (B)", "decode GB/token @ mlx-q4-g64",
                       "KV KiB/position fp16"]) + "\n")
    for model, hw in pairs:
        print(report(model, hw))


if __name__ == "__main__":
    main()
