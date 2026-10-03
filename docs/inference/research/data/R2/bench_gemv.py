# GEMV micro-benchmark for tinygrad on METAL: y[1,N] = x[1,K] @ W[N,K]^T
#   mode fp16 : W is a realized contiguous fp16 buffer (dense weights, "REALIZE=1" path)
#   mode q4_0 : W is the lazy GGUF Q4_0 decode expression over packed bytes (default tinygrad.llm path), cast to fp16
# Reports wall time per JIT call (median of REPS loops of ITERS calls, synchronized), and implied GB/s of weight bytes.
import sys, time, statistics, struct
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import ggml_data_to_tensor

mode, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ITERS, REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 50, 7
rng = np.random.default_rng(0)

if mode == "fp16":
  W = Tensor(rng.standard_normal((N, K), dtype=np.float32).astype(np.float16)).realize()
  wbytes = N*K*2
  def weight(): return W
elif mode in ("q4_0", "q4_0_f32"):
  nblk = N*K//32
  blocks = rng.integers(0, 256, size=(nblk, 18), dtype=np.uint8)
  blocks[:, 0:2] = np.frombuffer(np.full(nblk, 0.01, dtype=np.float16).tobytes(), dtype=np.uint8).reshape(nblk, 2)  # valid fp16 scales
  packed = Tensor(blocks.reshape(-1)).realize()
  wbytes = nblk*18
  dec = ggml_data_to_tensor(packed, N*K, 2).reshape(N, K)
  def weight(): return dec.cast(dtypes.float16) if mode == "q4_0" else dec
elif mode.startswith("ggml"):  # ggml<type id>, e.g. ggml8 = Q8_0, ggml14 = Q6_K; random packed bytes (timing only)
  from tinygrad.llm.gguf import _GGML_QUANT
  typ = int(mode[4:]); ne, nb = _GGML_QUANT[typ]
  nblk = N*K//ne
  blocks = rng.integers(0, 256, size=(nblk, nb), dtype=np.uint8)
  packed = Tensor(blocks.reshape(-1)).realize()
  wbytes = nblk*nb
  dec = ggml_data_to_tensor(packed, N*K, typ).reshape(N, K)
  def weight(): return dec.cast(dtypes.float16)
else: raise SystemExit("mode must be fp16|q4_0|q4_0_f32|ggml<type>")

x = Tensor(rng.standard_normal((1, K), dtype=np.float32).astype(np.float16)).realize()

@TinyJit
def f(x:Tensor) -> Tensor: return (x @ weight().T).realize()

t0 = time.perf_counter()
for i in range(3): f(x)   # cnt0 eager, cnt1 capture (+BEAM if JITBEAM/BEAM set), cnt2 replay
Device["METAL"].synchronize()
warm_s = time.perf_counter()-t0
times = []
for r in range(REPS):
  Device["METAL"].synchronize()
  st = time.perf_counter()
  for i in range(ITERS): f(x)
  Device["METAL"].synchronize()
  times.append((time.perf_counter()-st)/ITERS)
med = statistics.median(times)
print(f"RESULT mode={mode} N={N} K={K} warmup_s={warm_s:.2f} per_call_us_median={med*1e6:.1f} "
      f"min={min(times)*1e6:.1f} max={max(times)*1e6:.1f} weight_MB={wbytes/1e6:.2f} implied_GBps={wbytes/med/1e9:.1f}")
