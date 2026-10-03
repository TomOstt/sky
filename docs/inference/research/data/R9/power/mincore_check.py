# Report the fraction of a file's pages resident in the unified buffer cache, via mmap(PROT_READ)+mincore (no page touching).
import os, sys, ctypes as C, ctypes.util
libc = C.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = C.c_void_p
libc.mmap.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_long]
libc.munmap.argtypes = [C.c_void_p, C.c_size_t]
libc.mincore.argtypes = [C.c_void_p, C.c_size_t, C.c_char_p]
PROT_READ, MAP_SHARED = 1, 1
PAGE = os.sysconf("SC_PAGE_SIZE")
def resident_fraction(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
    if addr in (None, C.c_void_p(-1).value):
        raise OSError(C.get_errno(), "mmap failed")
    npages = (size + PAGE - 1) // PAGE
    vec = C.create_string_buffer(npages)
    if libc.mincore(C.c_void_p(addr), size, vec) != 0:
        raise OSError(C.get_errno(), "mincore failed")
    inc = sum(1 for b in vec.raw[:npages] if b & 1)
    libc.munmap(C.c_void_p(addr), size); os.close(fd)
    return inc / npages, npages
for p in sys.argv[1:]:
    fr, n = resident_fraction(p)
    print(f"{os.path.basename(p)}: resident {fr*100:.1f}% of {n} pages (page={PAGE})")
