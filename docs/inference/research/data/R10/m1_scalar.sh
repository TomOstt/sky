#!/bin/bash
# Group 1: scalar, calls, small vectors. Two sizes each, to separate fixed cost from per-unit cost.
T="python3 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/timeit.py -n 9 -w 2"
SKY=/Users/talperez/sky/target/release/sky
B=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/bin; V=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var; K=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky
echo "== floors"
for e in ast sir native auto; do $T --label "floor sky $e" -- $SKY run --engine=$e $V/floor.sky; done
echo "== fib 30 / 32"
$T --label "c fib30" -- $B/c_fib 30
$T --label "c fib32" -- $B/c_fib 32
$T --label "rust fib30" -- $B/rust_fib 30
$T --label "rust fib32" -- $B/rust_fib 32
$T --label "sky native fib30" -- $SKY run --engine=native $K/fib-30.sky
$T --label "sky native fib32" -- $SKY run --engine=native $V/fib-32.sky
echo "== nbody 1e5 / 4e5"
$T --label "c nbody1e5" -- $B/c_nbody 100000
$T --label "c nbody4e5" -- $B/c_nbody 400000
$T --label "rust nbody1e5" -- $B/rust_nbody 100000
$T --label "rust nbody4e5" -- $B/rust_nbody 400000
for slp in auto off always; do
SKY_SLP=$slp $T --label "sky native nbody1e5 SLP=$slp" -- $SKY run --engine=native $K/nbody.sky
SKY_SLP=$slp $T --label "sky native nbody4e5 SLP=$slp" -- $SKY run --engine=native $V/nbody-4e5.sky
done
echo "== mandelbrot maxit 500 / 2000"
$T --label "c mandel500" -- $B/c_mandelbrot 300 500
$T --label "c mandel2000" -- $B/c_mandelbrot 300 2000
$T --label "rust mandel500" -- $B/rust_mandelbrot 300 500
$T --label "rust mandel2000" -- $B/rust_mandelbrot 300 2000
$T --label "sky native mandel500" -- $SKY run --engine=native $K/mandelbrot.sky
$T --label "sky native mandel2000" -- $SKY run --engine=native $V/mandel-2000.sky
echo "== rk4 2e5 / 8e5"
$T --label "c rk4 2e5" -- $B/c_rk4 200000
$T --label "c rk4 8e5" -- $B/c_rk4 800000
$T --label "rust rk4 2e5" -- $B/rust_rk4 200000
$T --label "rust rk4 8e5" -- $B/rust_rk4 800000
$T --label "sky native rk4 2e5" -- $SKY run --engine=native $K/rk4.sky
$T --label "sky native rk4 8e5" -- $SKY run --engine=native $V/rk4-8e5.sky
