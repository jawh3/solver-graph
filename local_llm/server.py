#!/usr/bin/env python3
"""
local_llm/server.py — minimal OpenAI-compatible server for the laptop pilot.

The lab runs Qwen3.5-27B-FP8 on dual A5000s via vLLM. On an 8 GB laptop GPU that
model is impossible, so the pilot substitutes a small instruct model
(Qwen2.5-7B/3B-Instruct) loaded in 4-bit (bitsandbytes NF4) and serves it behind
the SAME OpenAI `/v1/chat/completions` API the real solver (`solver/solver.py`)
already speaks. Nothing in the solver changes — only `VLLM_URL`/`MODEL_NAME`.

This is a *disclosed model substitution* (documented in the paper's limitations):
the method is model-agnostic; the pilot demonstrates the learning dynamics and the
false-lesson suppression mechanisms, not the lab's absolute solve rate.

Run:
    python -m local_llm.server --model /path/to/Qwen2.5-7B-Instruct --port 8000
Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions.
Unknown request fields (e.g. the solver's `chat_template_kwargs`) are ignored.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

_state = {"model": None, "tok": None, "name": "local-model", "lock": threading.Lock()}


def load_model(path: str, dtype: str = "nf4"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(path)
    if dtype == "nf4":
        qc = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=qc, device_map="cuda", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    _state["model"], _state["tok"] = model, tok
    return tok, model


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": _state["name"], "object": "model"}]}


def _generate(messages, temperature: float, max_tokens: int, seed=None) -> str:
    tok, model = _state["tok"], _state["model"]
    if seed is not None:
        torch.manual_seed(int(seed))
    # Qwen3/3.5 are hybrid reasoning models; with a short tool-reply token budget,
    # thinking tokens would crowd out the JSON answer. Disable thinking for the
    # pilot (documented). Qwen2.5 tokenizers ignore the kwarg -> try/except.
    try:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False).to(model.device)
    except TypeError:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens, do_sample=do_sample,
        pad_token_id=tok.eos_token_id, use_cache=True)
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=0.9)
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)
    new_tokens = out[0][input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()                       # tolerate any extra fields
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.3)
    max_tokens = min(int(body.get("max_tokens", 512) or 512), 1536)
    seed = body.get("seed")
    t0 = time.time()
    # serialize generation: one GPU, one model
    with _state["lock"]:
        try:
            text = _generate(messages, temperature, max_tokens, seed)
        except Exception as e:  # surface as an OpenAI-style error
            return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
    # rough token accounting (chars/4) for RunRecord.tokens
    approx_tokens = max(1, len(text) // 4)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(t0),
        "model": _state["name"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": approx_tokens,
                  "total_tokens": approx_tokens},
        "_latency_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a local HF model dir")
    ap.add_argument("--name", default=None, help="model id advertised over the API")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dtype", default="nf4", choices=["nf4", "fp16"])
    args = ap.parse_args()
    _state["name"] = args.name or args.model.rstrip("/").split("/")[-1]
    print(f"[local_llm] loading {args.model} ({args.dtype}) ...", flush=True)
    load_model(args.model, dtype=args.dtype)
    print(f"[local_llm] ready: {_state['name']} on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
