import json, sys, transformers
from transformers import AutoConfig
print("transformers", transformers.__version__)
for name in ["llama-3.1-8b-instruct","qwen3-32b","gemma-3-27b-it","gpt-oss-20b","phi-4-mini-instruct","deepseek-v3"]:
    src = json.load(open(f"configs/{name}.json"))
    try:
        cfg = AutoConfig.for_model(**src) if "model_type" in src else None
    except Exception as e:
        print(name, "ERR", type(e).__name__, str(e)[:200]); continue
    d = cfg.to_dict()
    if "text_config" in d: d = d["text_config"]
    keys = {k: d.get(k, "<absent>") for k in ["rope_theta","rope_scaling","rope_parameters","layer_types","sliding_window_pattern","rope_local_base_freq","torch_dtype","dtype","tie_word_embeddings","head_dim"]}
    lt = keys["layer_types"]
    if isinstance(lt, list): keys["layer_types"] = f"list[{len(lt)}] first6={lt[:6]}"
    print("==", name); [print("   ", k, "=", json.dumps(v)[:300]) for k, v in keys.items()]
