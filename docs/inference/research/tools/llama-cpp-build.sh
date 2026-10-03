#!/bin/bash
# R5: rebuild llama.cpp with a workaround for a stale partial libc++ header dir in
# /Library/Developer/CommandLineTools/usr/include/c++/v1 (11 leftover entries from 2022/23
# CLT installs) that shadows the SDK's libc++ headers -> "'array' file not found".
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
cd $S/src/llama.cpp || exit 1
if [ -f $S/llama.done ]; then echo "llama.done already present; skipping"; exit 0; fi
SDK=$(xcrun --show-sdk-path)
FIX="-nostdinc++ -isystem $SDK/usr/include/c++/v1"
echo "[R5] reconfigure with CMAKE_CXX_FLAGS=$FIX at $(date)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS="$FIX" -DCMAKE_OBJCXX_FLAGS="$FIX" \
 && cmake --build build -j 10 --target llama-bench llama-cli llama-server llama-perplexity llama-speculative \
      llama-speculative-simple llama-lookup llama-batched-bench test-backend-ops llama-completion \
 && touch $S/llama.done && echo "[R5] build OK at $(date)"
