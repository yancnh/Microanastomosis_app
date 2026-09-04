"""FastAPI orchestrator for the live microvascular-bypass analysis dashboard."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


app = FastAPI(title="Microanastomosis Vision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
CONDA_EXE = os.getenv("CONDA_EXE", "conda")

DETECT_ENV = os.getenv("DETECT_ENV", "bypass")
DETECT_SCRIPT = os.getenv("DETECT_SCRIPT", str(APP_DIR / "detection_worker.py"))
DETECT_ARGS = os.getenv("DETECT_ARGS", "")
ACTION_ENV = os.getenv("ACTION_ENV", "transformer")
ACTION_SCRIPT = os.getenv("ACTION_SCRIPT", str(APP_DIR / "action_worker.py"))
ACTION_ARGS = os.getenv("ACTION_ARGS", "")
LLM_ENV = os.getenv("LLM_ENV", "localllm")
LLM_SCRIPT = os.getenv("LLM_SCRIPT", str(APP_DIR / "llm_worker.py"))
LLM_ARGS = os.getenv("LLM_ARGS", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen")

ACTION_FALLBACK_LABELS = [
    "No action",
    "Vessel cutting",
    "Needle handling",
    "Needle touching vessel",
    "Needle withdrawing",
    "Knot tying",
    "Knot cutting",
]
_last_tracks: dict[int, dict[str, float]] = {}
_state_lock = threading.Lock()


class JsonLineWorker:
    """Long-lived worker isolated in its original Conda environment."""

    def __init__(
        self,
        name: str,
        environment: str,
        script: str,
        extra_args: str = "",
        ready_timeout: float = 180.0,
    ) -> None:
        self.name = name
        self.environment = environment
        self.script = Path(script) if script else None
        self.extra_args = extra_args
        self.ready_timeout = ready_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._call_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._ready = False
        self._loading = False
        self._last_error = ""

    @property
    def configured(self) -> bool:
        return bool(self.script)

    @property
    def alive(self) -> bool:
        return bool(self._ready and self._proc is not None and self._proc.poll() is None)

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "ready": self.alive,
            "loading": self._loading,
            "environment": self.environment,
            "error": self._last_error or None,
        }

    def _command(self) -> list[str]:
        if not self.script or not self.script.is_file():
            raise FileNotFoundError(f"{self.name} worker not found: {self.script}")
        command = [
            CONDA_EXE,
            "run",
            "--no-capture-output",
            "-n",
            self.environment,
            "python",
            str(self.script),
        ]
        if self.extra_args:
            command.extend(shlex.split(self.extra_args))
        return command

    def ensure_started(self) -> bool:
        if self.alive:
            return True
        with self._start_lock:
            if self.alive:
                return True
            self._ready = False
            self._loading = True
            self._last_error = ""
            try:
                command = self._command()
                print(f"[{self.name}] Starting: {' '.join(command)}", file=sys.stderr)
                self._proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=sys.stderr,
                    text=True,
                    bufsize=1,
                )
                deadline = time.monotonic() + self.ready_timeout
                while time.monotonic() < deadline:
                    line = self._proc.stdout.readline() if self._proc.stdout else ""
                    if not line:
                        code = self._proc.poll()
                        raise RuntimeError(f"worker exited before ready (code {code})")
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if message.get("status") == "ready":
                        self._ready = True
                        print(f"[{self.name}] Ready", file=sys.stderr)
                        return True
                raise TimeoutError(f"worker did not become ready in {self.ready_timeout:.0f}s")
            except Exception as exc:
                self._last_error = str(exc)
                print(f"[{self.name}] Start failed: {exc}", file=sys.stderr)
                self.stop()
                return False
            finally:
                self._loading = False

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._call_lock:
            if not self.ensure_started():
                return {"error": self._last_error or f"{self.name} unavailable"}
            try:
                assert self._proc and self._proc.stdin and self._proc.stdout
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("worker closed stdout")
                result = json.loads(line)
                self._last_error = str(result.get("error") or "")
                return result
            except Exception as exc:
                self._last_error = str(exc)
                print(f"[{self.name}] Call failed: {exc}", file=sys.stderr)
                self.stop()
                return {"error": str(exc)}

    def stop(self) -> None:
        process = self._proc
        self._proc = None
        self._ready = False
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()


detect_worker = JsonLineWorker(
    "detection", DETECT_ENV, DETECT_SCRIPT, DETECT_ARGS, ready_timeout=180
)
action_worker = JsonLineWorker(
    "action", ACTION_ENV, ACTION_SCRIPT, ACTION_ARGS, ready_timeout=300
)
llm_worker = JsonLineWorker(
    "llm", LLM_ENV, LLM_SCRIPT, LLM_ARGS, ready_timeout=600
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
async def startup_event() -> None:
    # Vision workers are continuous; Qwen stays lazy until the first chat request.
    for worker in (detect_worker, action_worker):
        if worker.configured:
            threading.Thread(target=worker.ensure_started, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    for worker in (detect_worker, action_worker, llm_worker):
        worker.stop()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "online",
        "models": {
            "detection": detect_worker.status(),
            "action": action_worker.status(),
            "llm": {
                **llm_worker.status(),
                "configured": bool(LLM_API_URL or llm_worker.configured),
                "api_mode": bool(LLM_API_URL),
            },
        },
    }


@app.post("/api/detect-frame")
async def detect_frame(
    frame: UploadFile = File(...),
    source: str = Form("file"),
    timestamp_ms: float = Form(0),
    display_width: int = Form(0),
    display_height: int = Form(0),
) -> dict[str, Any]:
    """Run only the low-latency tracking path used by the live overlay."""
    started = time.perf_counter()
    suffix = Path(frame.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(
        prefix="vision-detect-", suffix=suffix, delete=False
    ) as temporary:
        frame_path = Path(temporary.name)
        temporary.write(await frame.read())
    try:
        detection = await run_in_threadpool(
            detect_worker.call, {"image": str(frame_path)}
        )
        if detection.get("error"):
            raise HTTPException(
                status_code=502, detail=f"Detection error: {detection['error']}"
            )
        tracks = add_kinematics(detection.get("tracks", []), timestamp_ms)
        return {
            "source": source,
            "tracks": tracks,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "image_width": detection.get("image_width") or display_width or 960,
            "image_height": detection.get("image_height") or display_height or 540,
            "model_status": {"detection": "ready"},
        }
    finally:
        frame_path.unlink(missing_ok=True)


@app.post("/api/action-frame")
async def action_frame(
    frame: UploadFile = File(...),
    source: str = Form("file"),
    timestamp_ms: float = Form(0),
) -> dict[str, Any]:
    """Run action segmentation independently from the live tracking path."""
    started = time.perf_counter()
    suffix = Path(frame.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(
        prefix="vision-action-", suffix=suffix, delete=False
    ) as temporary:
        frame_path = Path(temporary.name)
        temporary.write(await frame.read())
    try:
        result = await run_in_threadpool(
            action_worker.call,
            {"image": str(frame_path), "timestamp_ms": timestamp_ms},
        )
        if result.get("error"):
            raise HTTPException(
                status_code=502, detail=f"Action segmentation error: {result['error']}"
            )
        actions = result.get("actions") or fallback_actions()
        dominant = result.get("dominant_action") or max(
            actions, key=lambda item: float(item.get("confidence", 0))
        ).get("name", "Unknown")
        return {
            "source": source,
            "actions": actions,
            "dominant_action": dominant,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "model_status": {"action": "ready"},
        }
    finally:
        frame_path.unlink(missing_ok=True)


@app.post("/api/analyze-frame")
async def analyze_frame(
    frame: UploadFile = File(...),
    source: str = Form("file"),
    timestamp_ms: float = Form(0),
    display_width: int = Form(0),
    display_height: int = Form(0),
) -> dict[str, Any]:
    started = time.perf_counter()
    suffix = Path(frame.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(
        prefix="vision-frame-", suffix=suffix, delete=False
    ) as temporary:
        frame_path = Path(temporary.name)
        temporary.write(await frame.read())
    try:
        detection, action_result = await run_in_threadpool(
            _run_vision_models,
            frame_path,
            display_width,
            display_height,
            timestamp_ms,
        )
        tracks = add_kinematics(detection.get("tracks", []), timestamp_ms)
        actions = action_result.get("actions") or fallback_actions()
        dominant = action_result.get("dominant_action") or max(
            actions, key=lambda item: float(item.get("confidence", 0))
        ).get("name", "Unknown")
        return {
            "source": source,
            "tracks": tracks,
            "actions": actions,
            "dominant_action": dominant,
            "performance_score": estimate_performance(tracks, actions),
            "performance_method": "heuristic",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "image_width": detection.get("image_width") or display_width or 960,
            "image_height": detection.get("image_height") or display_height or 540,
            "model_status": {
                "detection": "error" if detection.get("error") else "ready",
                "action": "error" if action_result.get("error") else "ready",
            },
        }
    finally:
        frame_path.unlink(missing_ok=True)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    if LLM_API_URL:
        answer = await run_in_threadpool(call_llm_api, request.question, request.context)
        return {"answer": answer}
    result = await run_in_threadpool(
        llm_worker.call,
        {"question": request.question, "context": request.context},
    )
    if result.get("answer"):
        return {"answer": str(result["answer"])}
    raise HTTPException(status_code=502, detail=f"Local Qwen error: {result.get('error')}")


def _run_vision_models(
    frame_path: Path,
    display_width: int,
    display_height: int,
    timestamp_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    detection = detect_worker.call({"image": str(frame_path)})
    if detection.get("error"):
        detection.update(
            {
                "tracks": [],
                "image_width": display_width or 960,
                "image_height": display_height or 540,
            }
        )
    action = action_worker.call({"image": str(frame_path), "timestamp_ms": timestamp_ms})
    return detection, action


def fallback_actions() -> list[dict[str, Any]]:
    return [
        {"name": name, "confidence": 1.0 if index == 0 else 0.0, "class_id": index}
        for index, name in enumerate(ACTION_FALLBACK_LABELS)
    ]


def add_kinematics(
    tracks: list[dict[str, Any]], timestamp_ms: float
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with _state_lock:
        for item in tracks:
            track = dict(item)
            track_id = int(track.get("id", track.get("track_id", len(output) + 1)))
            bbox = track.get("bbox", [0, 0, 0, 0])
            centroid = track.get("centroid") or _box_centroid(bbox)
            previous = _last_tracks.get(track_id)
            speed = acceleration = heading = 0.0
            if previous and timestamp_ms > previous["timestamp_ms"]:
                dt = max((timestamp_ms - previous["timestamp_ms"]) / 1000.0, 1e-3)
                dx = float(centroid["x"]) - previous["x"]
                dy = float(centroid["y"]) - previous["y"]
                speed = math.hypot(dx, dy) / dt
                acceleration = (speed - previous["speed"]) / dt
                heading = math.degrees(math.atan2(dy, dx))
            _last_tracks[track_id] = {
                "x": float(centroid["x"]),
                "y": float(centroid["y"]),
                "speed": float(speed),
                "timestamp_ms": float(timestamp_ms),
            }
            track.update(
                {
                    "id": track_id,
                    "centroid": centroid,
                    "speed": float(track.get("speed", speed)),
                    "acceleration": float(track.get("acceleration", acceleration)),
                    "heading": float(track.get("heading", heading)),
                }
            )
            output.append(track)
    return output


def _box_centroid(bbox: Any) -> dict[str, float]:
    if isinstance(bbox, dict):
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        width = float(bbox.get("w", bbox.get("width", 0)))
        height = float(bbox.get("h", bbox.get("height", 0)))
        return {"x": x + width / 2, "y": y + height / 2}
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2}


def estimate_performance(
    tracks: list[dict[str, Any]], actions: list[dict[str, Any]]
) -> int:
    """Dashboard heuristic only; this is not a validated clinical score."""
    if not tracks:
        return 40
    mean_confidence = sum(float(item.get("confidence", 0.5)) for item in tracks) / len(tracks)
    mean_speed = sum(float(item.get("speed", 0)) for item in tracks) / len(tracks)
    action_confidence = max(
        (float(item.get("confidence", 0)) for item in actions), default=0.5
    )
    value = 45 + mean_confidence * 30 + action_confidence * 20 - min(mean_speed / 60, 10)
    return int(max(0, min(100, round(value))))


def call_llm_api(question: str, context: dict[str, Any]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise surgical video analyst. Use only the supplied "
                "tracking, kinematics, action, and heuristic performance context."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "context": context}, ensure_ascii=False
            ),
        },
    ]
    try:
        response = requests.post(
            LLM_API_URL,
            json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2},
            timeout=180,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}") from exc
