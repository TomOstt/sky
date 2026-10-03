# Entropy of MLX-style affine g64 4-bit / 3-bit codes for Gaussian, Laplace and Student-t(4) weights (pure Python).
import random, math, collections
random.seed(0)
def laplace(): 
    u = random.random() - 0.5
    return -math.copysign(1, u) * math.log(1 - 2 * abs(u))
def student4():
    z = random.gauss(0, 1); v = sum(random.gauss(0, 1) ** 2 for _ in range(4)) / 4
    return z / math.sqrt(v)
def sim(gen, bits, G=64, ngroups=20000):
    L = 2 ** bits - 1; hist = collections.Counter()
    for _ in range(ngroups):
        w = [gen() for _ in range(G)]
        lo, hi = min(w), max(w); s = (hi - lo) / L
        for x in w: hist[min(L, max(0, round((x - lo) / s)))] += 1
    n = sum(hist.values())
    return -sum(c / n * math.log2(c / n) for c in hist.values())
for name, g in (("gaussian", lambda: random.gauss(0, 1)), ("laplace", laplace), ("student-t4", student4)):
    print(f"{name:11s} 4-bit g64 code entropy = {sim(g,4):.3f} bits (of 4) ; 3-bit g64 = {sim(g,3):.3f} (of 3)")
