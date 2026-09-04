"""Persistent local Qwen chat worker using a JSON-lines stdin/stdout protocol."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a concise surgical video analyst for microvascular bypass training. "
    "Use only the supplied live dashboard context. Distinguish observations from "
    "inferences, state when data is unavailable, and do not give clinical advice."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": request.get("question", "Summarize the scene."),
                            "context": request.get("context", {}),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, inputs["input_ids"].shape[1] :]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            result = {"answer": answer}
        except Exception as exc:
            print(f"[llm_worker] {exc}", file=sys.stderr)
            result = {"error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
