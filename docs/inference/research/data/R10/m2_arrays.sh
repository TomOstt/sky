#!/bin/bash
T="python3 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/timeit.py -n 7 -w 2"
SKY=/Users/talperez/sky/target/release/sky
B=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/bin; V=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var; K=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky
echo "== elementwise 1e7: R=2 / R=6"
$T --label "c elem1e7 r2" -- $B/c_elementwise 1000 10000 2
$T --label "c elem1e7 r6" -- $B/c_elementwise 1000 10000 6
for e in native sir; do
$T --label "sky $e elem1e7 r2" -- $SKY run --engine=$e $K/elementwise-1e7.sky
$T --label "sky $e elem1e7 r6" -- $SKY run --engine=$e $V/elem1e7-r6.sky
done
$T --label "sky native T1 elem1e7 r2" -- $SKY run --engine=native --threads=1 $K/elementwise-1e7.sky
$T --label "sky native T1 elem1e7 r6" -- $SKY run --engine=native --threads=1 $V/elem1e7-r6.sky
$T --label "sky sir T1 elem1e7 r2" -- $SKY run --engine=sir --threads=1 $K/elementwise-1e7.sky
$T --label "sky sir T1 elem1e7 r6" -- $SKY run --engine=sir --threads=1 $V/elem1e7-r6.sky
echo "== elementwise 1e6: R=20 / R=60"
$T --label "c elem1e6 r20" -- $B/c_elementwise 1000 1000 20
$T --label "c elem1e6 r60" -- $B/c_elementwise 1000 1000 60
for e in native sir; do
$T --label "sky $e elem1e6 r20" -- $SKY run --engine=$e $K/elementwise-1e6.sky
$T --label "sky $e elem1e6 r60" -- $SKY run --engine=$e $V/elem1e6-r60.sky
$T --label "sky $e T1 elem1e6 r20" -- $SKY run --engine=$e --threads=1 $K/elementwise-1e6.sky
$T --label "sky $e T1 elem1e6 r60" -- $SKY run --engine=$e --threads=1 $V/elem1e6-r60.sky
done
echo "== linear_chain 1e6: R=20 / R=60"
$T --label "c lin1e6 r20" -- $B/c_linear_chain 1000 1000 20
$T --label "c lin1e6 r60" -- $B/c_linear_chain 1000 1000 60
for e in native sir; do
$T --label "sky $e lin1e6 r20" -- $SKY run --engine=$e $K/linear_chain-1e6.sky
$T --label "sky $e lin1e6 r60" -- $SKY run --engine=$e $V/lin1e6-r60.sky
done
echo "== matmul 512: R=2 / R=6"
$T --label "c matmul512 r2" -- $B/c_matmul 512 2
$T --label "c matmul512 r6" -- $B/c_matmul 512 6
$T --label "lib matmul512 r2" -- $B/lib_matmul 512 2
$T --label "lib matmul512 r6" -- $B/lib_matmul 512 6
$T --label "sky native matmul512 r2" -- $SKY run --engine=native $K/matmul-512.sky
$T --label "sky native matmul512 r6" -- $SKY run --engine=native $V/matmul512-r6.sky
$T --label "sky native T1 matmul512 r2" -- $SKY run --engine=native --threads=1 $K/matmul-512.sky
$T --label "sky native T1 matmul512 r6" -- $SKY run --engine=native --threads=1 $V/matmul512-r6.sky
$T --label "sky native relaxed matmul512 r2" -- $SKY run --engine=native --numerics=relaxed $K/matmul-512.sky
$T --label "sky native relaxed matmul512 r6" -- $SKY run --engine=native --numerics=relaxed $V/matmul512-r6.sky
echo "== cholesky 32: R=5000 / R=20000"
$T --label "c chol32 r5000" -- $B/c_cholesky 32 5000
$T --label "c chol32 r20000" -- $B/c_cholesky 32 20000
$T --label "lib chol32 r5000" -- $B/lib_cholesky 32 5000
$T --label "lib chol32 r20000" -- $B/lib_cholesky 32 20000
$T --label "sky native chol32 r5000" -- $SKY run --engine=native $K/cholesky-32.sky
$T --label "sky native chol32 r20000" -- $SKY run --engine=native $V/chol32-r20000.sky
$T --label "sky native relaxed chol32 r5000" -- $SKY run --engine=native --numerics=relaxed $K/cholesky-32.sky
$T --label "sky native relaxed chol32 r20000" -- $SKY run --engine=native --numerics=relaxed $V/chol32-r20000.sky
