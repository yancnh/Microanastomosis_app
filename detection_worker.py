"""Persistent YOLO + DeepSORT JSON-lines worker for the vision dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import detection_runner_for_app as runner
from tip_localizer import find_tip_in_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="tool_detect.yaml")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class Detector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(args.model)
        if args.fuse:
            self.model.fuse()
        runner.init_tracker()
        self.previous_bgr: np.ndarray | None = None
        self.tip_trails: dict[int, deque[dict[str, float]]] = {}
        self.last_tips: dict[int, np.ndarray] = {}
        self.tip_samples: dict[int, deque[np.ndarray]] = {}
        self.last_box_centers: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        runner.init_tracker()
        self.previous_bgr = None
        self.tip_trails.clear()
        self.last_tips.clear()
        self.tip_samples.clear()
        self.last_box_centers.clear()

    def analyze(self, image_path: str) -> dict:
        path = Path(image_path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not decode frame: {path}")
        height, width = frame.shape[:2]
        result = self.model.predict(
            source=str(path),
            data=self.args.data,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            device=self.device,
            half=bool(self.args.half and torch.cuda.is_available()),
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        detections = [
            (box, float(confidence), int(class_id))
            for box, confidence, class_id in zip(boxes, confidences, class_ids)
        ]

        try:
            deep_tracks = runner._tracker.update_tracks(detections, frame=frame) if detections else []
        except Exception as exc:
            print(f"[detection_worker] DeepSORT update skipped: {exc}", file=sys.stderr)
            deep_tracks = []

        tracked_boxes, identities, tracked_classes, tracked_confidences = [], [], [], []
        for track in deep_tracks:
            tracked_boxes.append(track.to_tlbr())
            identities.append(track.track_id)
            tracked_classes.append(track.det_class)
            tracked_confidences.append(track.det_conf)
        boxes_out, ids_out, classes_out, confs_out = runner.create_unique_class_boxes(
            tracked_boxes,
            identities,
            tracked_classes,
            tracked_confidences,
            detections,
        )

        motion_mask = None
        if runner._TIPS_AVAILABLE and self.previous_bgr is not None:
            motion_mask = runner._frame_diff_mask(self.previous_bgr, frame)
        names = result.names
        tracks = []
        for box, track_id, class_id, confidence in zip(
            boxes_out, ids_out, classes_out, confs_out
        ):
            x1, y1, x2, y2 = (float(value) for value in box)
            identifier = int(track_id)
            class_index = int(class_id)
            tip = None
            if motion_mask is not None:
                raw_tip = find_tip_in_roi(
                    frame,
                    motion_mask,
                    np.array([int(x1), int(y1), int(x2), int(y2)]),
                    self.last_tips.get(identifier),
                )
                if raw_tip is not None:
                    raw_tip = np.asarray(raw_tip, dtype=np.float32)
                    samples = self.tip_samples.setdefault(
                        identifier, deque(maxlen=5)
                    )
                    samples.append(raw_tip)
                    box_center = np.array(
                        [(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32
                    )
                    previous_box_center = self.last_box_centers.get(identifier)
                    self.last_box_centers[identifier] = box_center
                    previous_tip = self.last_tips.get(identifier)
                    if previous_tip is not None:
                        diagonal = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
                        box_shift = (
                            float("inf")
                            if previous_box_center is None
                            else float(np.linalg.norm(box_center - previous_box_center))
                        )
                        stationary = box_shift < max(1.5, 0.025 * diagonal)
                        if stationary:
                            recent = np.stack(samples)
                            candidate = np.median(recent, axis=0)
                            residual = float(np.linalg.norm(candidate - previous_tip))
                            if residual <= max(1.5, 0.012 * diagonal):
                                raw_tip = previous_tip
                            else:
                                recent_three = recent[-3:]
                                median_three = np.median(recent_three, axis=0)
                                spread = float(
                                    np.max(np.linalg.norm(recent_three - median_three, axis=1))
                                )
                                confirmed = (
                                    len(recent_three) == 3
                                    and spread < max(2.5, 0.035 * diagonal)
                                )
                                alpha = 0.55 if confirmed else 0.10
                                raw_tip = alpha * candidate + (1.0 - alpha) * previous_tip
                        else:
                            raw_tip = 0.82 * raw_tip + 0.18 * previous_tip
                    self.last_tips[identifier] = raw_tip
                    tip = {"x": float(raw_tip[0]), "y": float(raw_tip[1])}
            trail = self.tip_trails.setdefault(identifier, deque(maxlen=64))
            if tip is not None:
                trail.append(tip)
            confidence_value = float(confidence) if confidence is not None else 0.0
            if not np.isfinite(confidence_value):
                confidence_value = 0.0
            tracks.append(
                {
                    "id": identifier,
                    "label": str(names[class_index]),
                    "class_id": class_index,
                    "confidence": confidence_value,
                    "bbox": [x1, y1, x2, y2],
                    "centroid": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
                    "tip": tip,
                    "tip_trail": list(trail),
                }
            )
        self.previous_bgr = frame.copy()
        return {"tracks": tracks, "image_width": width, "image_height": height}


def main() -> None:
    detector = Detector(parse_args())
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "reset":
                detector.reset()
                response = {"status": "reset"}
            else:
                response = detector.analyze(request["image"])
        except Exception as exc:
            print(f"[detection_worker] {exc}", file=sys.stderr)
            response = {"error": str(exc)}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
