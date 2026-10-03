# Profile one tinygrad decode step for a local GGUF model. Run with DEBUG=2 so the JIT batch
# is compiled with per-kernel timestamps; prints a marker before the final token so the
# per-kernel "*** METAL" lines of exactly one decode step can be parsed from the log.
import sys, time
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.model import Transformer

path, ntok = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
st = time.perf_counter()
model, kv = Transformer.from_gguf(Tensor.from_url(path) if path.startswith("http") else path, max_context=1024)
print(f"LOAD_S {time.perf_counter()-st:.3f}", flush=True)
gen = model.generate([0])
for i in range(ntok):
  GlobalCounters.reset()
  if i == ntok-1: print("=====FINAL_TOKEN=====", flush=True)
  st = time.perf_counter()
  next(gen)
  et = time.perf_counter()-st
  print(f"TOKEN {i} wall_ms {et*1e3:.2f} kernels {GlobalCounters.kernel_count} global_mem_MB {GlobalCounters.global_mem/1e6:.1f} "
        f"gpu_sum_ms {GlobalCounters.time_sum_s*1e3:.2f}", flush=True)
