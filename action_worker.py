from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_LABELS = [
    "No action",
    "Vessel cutting",
    "Needle handling",
    "Needle touching vessel",
    "Needle withdrawing",
    "Knot tying",
    "Knot cutting",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ActSegFormer .pth checkpoint")
    parser.add_argument(
        "--inference-script",
        default=(
            
        ),
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--short-side-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--labels", default="|".join(DEFAULT_LABELS))
    return parser.parse_args()


def import_inference_module(path: str):
    script = Path(path)
    if not script.is_file():
        raise FileNotFoundError(f"Model inference script not found: {script}")
    spec = importlib.util.spec_from_file_location("inference", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split("|")]
    if len(labels) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} labels, got {len(labels)}")

    module = import_inference_module(args.inference_script)
    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = module.load_model(
        args.model,
        device,
        model_name="ActSegFormer",
        num_classes=args.num_classes,
        num_frames=args.num_frames,
    )
    resize, transform = module._build_transforms(args.short_side_size)
    frames: deque[np.ndarray] = deque(maxlen=args.num_frames)
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "reset":
                frames.clear()
                result = {"status": "reset"}
            else:
                frame = cv2.imread(request["image"])
                if frame is None:
                    raise ValueError(f"Could not decode frame: {request['image']}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Give the model its trained 16-frame input immediately. During
                # warmup, repeat the earliest available frame on the left.
                clip_frames = list(frames)
                clip_frames = [clip_frames[0]] * (args.num_frames - len(clip_frames)) + clip_frames
                resized = resize(clip_frames)
                clip = transform(np.stack(resized, axis=0)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    autocast_enabled = device.type == "cuda"
                    with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                        logits = model(clip)
                    probs = torch.softmax(logits, dim=-1)
                    current = probs[0, -1] if probs.ndim == 3 else probs[0]
                scores = current.float().cpu().tolist()
                actions = [
                    {"name": name, "confidence": float(score), "class_id": index}
                    for index, (name, score) in enumerate(zip(labels, scores))
                ]
                best = max(range(len(scores)), key=scores.__getitem__)
                result = {
                    "actions": actions,
                    "dominant_action": labels[best],
                    "frames_seen": len(frames),
                    "window_size": args.num_frames,
                }
        except Exception as exc:
            print(f"[action_worker] {exc}", file=sys.stderr)
            result = {"error": str(exc)}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
