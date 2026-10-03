import json, os, glob, sys, copy, traceback, difflib
import minijinja
from transformers.utils.chat_template_utils import render_jinja_template
T = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(T, "repos")
Q = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/*"))[0]
REPOS = {
 "Qwen3-8B": Q,
 "Qwen2.5-7B-Instruct": f"{R}/Qwen__Qwen2.5-7B-Instruct",
 "Llama-3.1-8B-Instruct": f"{R}/unsloth__Llama-3.1-8B-Instruct",
 "gemma-3-4b-it": f"{R}/unsloth__gemma-3-4b-it",
 "Mistral-7B-Instruct-v0.3": f"{R}/mistralai__Mistral-7B-Instruct-v0.3",
 "Mistral-Small-3.2-24B": f"{R}/unsloth__Mistral-Small-3.2-24B-Instruct-2506",
 "phi-4": f"{R}/microsoft__phi-4",
 "DeepSeek-R1-Distill-Qwen-7B": f"{R}/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B",
 "DeepSeek-R1-Distill-Llama-8B": f"{R}/deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
 "gpt-oss-20b": f"{R}/openai__gpt-oss-20b",
}
def load_template(d, tools=False):
    j = os.path.join(d, "chat_template.jinja")
    if os.path.exists(j): return open(j).read(), "chat_template.jinja"
    c = json.load(open(os.path.join(d, "tokenizer_config.json")))
    t = c.get("chat_template")
    if isinstance(t, list):
        m = {x["name"]: x["template"] for x in t}
        if tools and "tool_use" in m: return m["tool_use"], "tokenizer_config[tool_use]"
        return m["default"], "tokenizer_config[default]"
    return t, "tokenizer_config.json"
def special_map(d):
    c = json.load(open(os.path.join(d, "tokenizer_config.json")))
    out = {}
    for k in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"]:
        v = c.get(k)
        if isinstance(v, dict): v = v.get("content")
        if v is not None: out[k] = v
    if c.get("additional_special_tokens"): out["additional_special_tokens"] = c["additional_special_tokens"]
    return out
FIXED_DATE = lambda fmt: "01 Oct 2026" if "%b" in fmt else "2026-10-01"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather <for> a city & date",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Zürich"},
          "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
CONV = {
 "basic": dict(messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi there!"},
                         {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "2+2? Answer with “one” word — please."}],
               add_generation_prompt=True),
 "tools": dict(messages=[{"role": "system", "content": "You can call tools."}, {"role": "user", "content": "Weather in Zürich?"},
                         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}}]},
                         {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\": 21}"}],
               tools=TOOLS, add_generation_prompt=True),
 "reasoning_history": dict(messages=[{"role": "user", "content": "Q1"},
                         {"role": "assistant", "content": "<think>\nsecret chain\n</think>\n\nA1"},
                         {"role": "user", "content": "Q2"}], add_generation_prompt=True),
}
def py_tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(msg): raise minijinja.TemplateError(msg) if hasattr(minijinja,'TemplateError') else Exception(msg)
def mj_render(tpl, ctx, pycompat, tojson_override):
    env = minijinja.Environment(trim_blocks=True, lstrip_blocks=True, pycompat=pycompat)
    env.add_function("raise_exception", raise_exception)
    env.add_function("strftime_now", FIXED_DATE)
    if tojson_override: env.add_filter("tojson", py_tojson)
    env.add_template("t", tpl)
    return env.render_template("t", **ctx)
results = {}
for model, d in REPOS.items():
    for cname, conv in CONV.items():
        variants = [{}]
        if model == "Qwen3-8B": variants = [{"enable_thinking": True}, {"enable_thinking": False}]
        for extra in variants:
            key = f"{model} | {cname} | {extra or ''}"
            tpl, src = load_template(d, tools="tools" in conv)
            ctx = dict(special_map(d)); ctx.update(copy.deepcopy(conv)); ctx.update(extra); ctx.setdefault("tools", None); ctx.setdefault("documents", None)
            r = {"template_src": src}
            try:
                kw = {k: v for k, v in ctx.items() if k not in ("messages", "tools", "documents", "add_generation_prompt")}
                kw["strftime_now"] = FIXED_DATE
                ref = render_jinja_template(conversations=[ctx["messages"]], tools=ctx["tools"], documents=None, chat_template=tpl,
                                            add_generation_prompt=ctx["add_generation_prompt"], **kw)[0][0]
                r["jinja2"] = ref
            except Exception as e:
                r["jinja2_err"] = f"{type(e).__name__}: {str(e)[:200]}"; ref = None
            for pyc in (True, False):
                for tj in (False, True):
                    tag = f"minijinja(pycompat={pyc},tojson={'py' if tj else 'builtin'})"
                    try:
                        out = mj_render(tpl, copy.deepcopy(ctx), pyc, tj)
                        if ref is None: r[tag] = "rendered (jinja2 failed)"
                        elif out == ref: r[tag] = "IDENTICAL"
                        else:
                            sm = difflib.SequenceMatcher(None, ref, out)
                            i = next((a.size for a in sm.get_matching_blocks()), 0)
                            first = next(i for i,(x,y) in enumerate(zip(ref+"\0", out+"\0")) if x != y)
                            r[tag] = f"DIFF at char {first}: jinja2={ref[max(0,first-30):first+50]!r} | minijinja={out[max(0,first-30):first+50]!r}"
                    except Exception as e:
                        r[tag] = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}"
            results[key] = r
json.dump(results, open(os.path.join(T, "tpl_results.json"), "w"), ensure_ascii=False, indent=1)
for k, r in results.items():
    print("==", k, "[", r["template_src"], "]")
    if "jinja2_err" in r: print("   jinja2 ERROR:", r["jinja2_err"])
    for t, v in r.items():
        if t.startswith("minijinja"): print("  ", t, ":", v if len(str(v)) < 400 else str(v)[:400])
