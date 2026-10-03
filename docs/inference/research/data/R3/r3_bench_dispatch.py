# E. Dispatch overhead: evaluate N tiny dependent ops (x = x + 1 on a 1K vector)
# in one graph, vs one op. Also: graph-build-only time, N independent ops,
# and the mx.compile'd (fused) chain.
# Usage: python r3_bench_dispatch.py
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import env_header, med

env_header("E_dispatch")
x = mx.ones((1024,), dtype=mx.float16)
mx.eval(x)


def chain(x, n):
    y = x
    for _ in range(n):
        y = y + 1
    return y


def timeit(fn, reps):
    for _ in range(max(3, reps // 10)):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return med(ts)


rows = []
for n in [1, 10, 100, 1000, 3000]:
    reps = 200 if n <= 100 else 30
    # build only (no eval): python + C++ graph construction cost
    t_build = timeit(lambda: chain(x, n), reps)
    # build + eval dependent chain
    t_dep = timeit(lambda: mx.eval(chain(x, n)), reps)
    # independent ops: n separate outputs from x, evaluated together
    t_ind = timeit(lambda: mx.eval([x + i for i in range(n)]), reps)
    # compiled (fused) chain
    cf = mx.compile(lambda a, n=n: chain(a, n))
    mx.eval(cf(x))
    t_cmp = timeit(lambda: mx.eval(cf(x)), reps)
    rows.append((n, t_build, t_dep, t_ind, t_cmp))
    print(f"N={n:5d}  build-only {t_build*1e6:9.1f} us ({t_build/n*1e6:6.2f} us/op) | "
          f"dep eval {t_dep*1e6:9.1f} us ({t_dep/n*1e6:6.2f} us/op) | "
          f"indep eval {t_ind*1e6:9.1f} us ({t_ind/n*1e6:6.2f} us/op) | "
          f"compiled {t_cmp*1e6:9.1f} us", flush=True)

t1 = rows[0][2]
for n, tb, td, ti, tc in rows[1:]:
    print(f"marginal dependent: N={n:5d}  (t(N)-t(1))/(N-1) = {(td - t1)/(n-1)*1e6:6.2f} us/op ;"
          f" minus build: {((td - tb) - (t1 - rows[0][1]))/(n-1)*1e6:6.2f} us/op")

# Pure eval of a pre-built graph is not possible to separate without re-building,
# so also measure: async_eval pipelining of 1000 single-op evals.
def many_small_evals(k):
    y = x
    for _ in range(k):
        y = y + 1
        mx.eval(y)
    return y

t = timeit(lambda: many_small_evals(100), 20)
print(f"100 x (op + eval):            {t*1e6:9.1f} us  -> {t/100*1e6:6.2f} us per synchronous eval")

def many_async(k):
    y = x
    for _ in range(k):
        y = y + 1
        mx.async_eval(y)
    mx.eval(y)
    return y

t = timeit(lambda: many_async(100), 20)
print(f"100 x (op + async_eval) + eval: {t*1e6:9.1f} us  -> {t/100*1e6:6.2f} us per async eval")
