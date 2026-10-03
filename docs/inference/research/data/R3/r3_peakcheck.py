# Check what mx.full / reductions do to get_peak_memory (no timing).
import mlx.core as mx
GB = 1e9
n = (512 << 20) // 2  # 512 MiB fp16
mx.reset_peak_memory()
x = mx.full((n,), 0.5, dtype=mx.float16); mx.eval(x)
print("after full: active %.3f peak %.3f cache %.3f" % (mx.get_active_memory()/GB, mx.get_peak_memory()/GB, mx.get_cache_memory()/GB))
mx.reset_peak_memory()
s = mx.sum(x); mx.eval(s)
print("after sum:  active %.3f peak %.3f cache %.3f" % (mx.get_active_memory()/GB, mx.get_peak_memory()/GB, mx.get_cache_memory()/GB))
mx.reset_peak_memory()
y = x * 2; mx.eval(y); del y
print("after x*2:  active %.3f peak %.3f cache %.3f" % (mx.get_active_memory()/GB, mx.get_peak_memory()/GB, mx.get_cache_memory()/GB))
mx.reset_peak_memory()
z = mx.zeros((n,), dtype=mx.float16); mx.eval(z)
print("after zeros:active %.3f peak %.3f cache %.3f" % (mx.get_active_memory()/GB, mx.get_peak_memory()/GB, mx.get_cache_memory()/GB))
