# Mandelbrot escape counts on a W x W grid over [-2, 1] x [-1.5, 1.5].
# usage: mandelbrot.py W MAXIT
import sys


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    total = 0
    for py in range(w):
        ci = -1.5 + py * (3.0 / w)
        for px in range(w):
            cr = -2.0 + px * (3.0 / w)
            zr = 0.0
            zi = 0.0
            i = 0
            while i < maxit:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                i += 1
            total += i
    print(total)


main()
