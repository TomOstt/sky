# tinygrad dispatch-overhead micro-benchmarks on METAL (medians of 5 reps).
#  jit1   : TinyJit with 1 tiny kernel                         -> per-call round trip
#  jit64  : TinyJit with a chain of 64 tiny kernels (1 batch)   -> (jit64-jit1)/63 = marginal cost per kernel in one ICB batch
#  eager1 : no JIT, (x+1).realize() each call                   -> trace + schedule-cache hit + lowering caches + link + run
#  sched  : time to build + schedule (cache miss) a fresh 64-op elementwise chain with distinct consts (compile-time cost)
import time, statistics
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context

dev = Device["METAL"]
x = Tensor.ones(16).contiguous().realize()

def med(fn, iters, reps=5):
  ts = []
  for _ in range(reps):
    dev.synchronize(); st = time.perf_counter()
    for _ in range(iters): fn()
    dev.synchronize(); ts.append((time.perf_counter()-st)/iters)
  return statistics.median(ts)

@TinyJit
def jit1(x): return (x+1).realize()
@TinyJit
def jit64(x):
  y = x
  for i in range(64): y = (y+1).contiguous()
  return y.realize()
for _ in range(3): jit1(x); jit64(x)
t1, t64 = med(lambda: jit1(x), 200), med(lambda: jit64(x), 100)
te = med(lambda: (x+1).realize(), 100)
print(f"RESULT jit1_us={t1*1e6:.1f} jit64_us={t64*1e6:.1f} marginal_per_kernel_us={(t64-t1)/63*1e6:.2f} eager1_us={te*1e6:.1f}")

# compile-time: schedule + codegen + compile a never-seen 64-kernel chain (distinct consts defeat the caches)
for trial in range(3):
  st = time.perf_counter()
  y = x
  for i in range(64): y = (y + (1000.0*trial + i + 0.5)).contiguous()
  t_build = time.perf_counter()
  y.realize()
  t_end = time.perf_counter()
  print(f"RESULT cold64 trial={trial} build_ms={(t_build-st)*1e3:.2f} realize_ms={(t_end-t_build)*1e3:.1f} per_kernel_ms={(t_end-t_build)*1e3/64:.2f}")
