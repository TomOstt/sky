# R4 shared helpers (mlx-lm measurements).
import ctypes, os, platform, statistics, subprocess, json, time

S = "/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad"
SRC = S + "/src"

def model_path(name):
    import glob
    return glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--mlx-community--{name}/snapshots/*/"))[0]

# ---- OS-level memory footprint (macOS proc_pid_rusage, RUSAGE_INFO_V4) ----
class _RUsageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(35)]

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")

def rusage():
    r = _RUsageV4()
    rc = _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r))
    if rc != 0:
        return {}
    return dict(phys_footprint=r.f7, resident=r.f6, wired=r.f5,
                lifetime_max_phys_footprint=r.f28, billed_energy_nj=r.f31,
                serviced_energy_nj=r.f32, interval_max_phys_footprint=r.f33)

# ---- prompt material: real text from local docs, truncated to exact token counts ----
def filler_text():
    parts = []
    for p in ("mlx-lm/mlx_lm/LORA.md", "mlx-lm/README.md", "mlx-lm/mlx_lm/SERVER.md",
              "mlx/README.md", "mlx-lm/mlx_lm/MANAGE.md", "mlx-lm/mlx_lm/LEARNED_QUANTS.md"):
        fp = os.path.join(SRC, p)
        if os.path.exists(fp):
            parts.append(open(fp).read())
    return "\n\n".join(parts)

def prompt_ids(tokenizer, n):
    ids = tokenizer.encode(filler_text())
    assert len(ids) >= n, (len(ids), n)
    return ids[:n]

def med(xs):
    return statistics.median(xs)

def env_info():
    import mlx.core as mx, importlib.metadata as md
    return dict(mlx=md.version("mlx"), mlx_lm=md.version("mlx-lm"),
                device=mx.device_info(), macos=platform.mac_ver()[0],
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
