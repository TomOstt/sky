// Mandelbrot escape counts on a W x W grid over [-2, 1] x [-1.5, 1.5].
// usage: mandelbrot W MAXIT
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long w = argc > 1 ? atol(argv[1]) : 300;
    long maxit = argc > 2 ? atol(argv[2]) : 500;
    long total = 0;
    for (long py = 0; py < w; py++) {
        double ci = -1.5 + py * (3.0 / w);
        for (long px = 0; px < w; px++) {
            double cr = -2.0 + px * (3.0 / w);
            double zr = 0.0, zi = 0.0;
            long i = 0;
            while (i < maxit) {
                double zr2 = zr * zr;
                double zi2 = zi * zi;
                if (zr2 + zi2 > 4.0) break;
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                i++;
            }
            total += i;
        }
    }
    printf("%ld\n", total);
    return 0;
}
