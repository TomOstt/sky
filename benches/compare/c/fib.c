// Naive doubly recursive Fibonacci. usage: fib N
#include <stdio.h>
#include <stdlib.h>

static long fib(long n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 30;
    printf("%ld\n", fib(n));
    return 0;
}
