cd "$(dirname "$0")"
./accdet 64 >/dev/null  # warm first-exec
for n in 64 512 1024; do
  for t in default 1 2 4 8; do
    for rep in 1 2 3; do
      if [ "$t" = default ]; then out=$(./accdet $n); else out=$(VECLIB_MAXIMUM_THREADS=$t ./accdet $n); fi
      echo "n=$n threads=$t rep=$rep :: $(echo $out | tr '\n' ' ')"
    done
  done
done
