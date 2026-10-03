#!/bin/bash
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
set -x
uv venv -q $S/venv --python 3.12 && uv pip install -q --python $S/venv/bin/python mlx mlx-lm huggingface_hub hf_transfer
$S/venv/bin/python -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, 'mlx_lm', mlx_lm.__version__); print(mx.metal.device_info())"
touch $S/venv.done
# llama.cpp (Metal, shaders embedded and compiled at runtime)
( cd $S/src/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > $S/llama-build.log 2>&1 && cmake --build build -j 8 --target llama-bench llama-cli llama-server llama-perplexity llama-speculative >> $S/llama-build.log 2>&1 && touch $S/llama.done ) &
# models (MLX in the HF cache; GGUF in scratchpad)
$S/venv/bin/hf download mlx-community/Qwen3-0.6B-4bit > /dev/null && echo ok-0.6b-mlx
$S/venv/bin/hf download mlx-community/Qwen3-8B-4bit > /dev/null && echo ok-8b-mlx
$S/venv/bin/hf download unsloth/Qwen3-0.6B-GGUF --include "*Q4_0.gguf" --local-dir $S/models > /dev/null && echo ok-0.6b-gguf
$S/venv/bin/hf download unsloth/Qwen3-8B-GGUF --include "*Q4_0.gguf" --local-dir $S/models > /dev/null && echo ok-8b-gguf
ls -la $S/models; du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3-* 2>/dev/null
wait
touch $S/setup.done
