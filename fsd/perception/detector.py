"""Object detector: YOLO on the GPU.

* If models/erlc.pt exists (made by Training mode from your own drives) it is
  used: it knows ER:LC-specific classes such as light backs and cones.
* Otherwise the stock COCO YOLO model is used. It already recognises cars,
  trucks, buses, people, traffic lights and stop signs, which covers most of
  ER:LC out of the box.
* If torch/ultralytics are missing, a colour-based fallback still finds lit
  traffic lights and stop signs (see lights.py / signs.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..world import Obj

# Classes of the custom ER:LC model (order matters: it is the training label order)
ERLC_CLASSES = ["car", "person", "cone", "light_red", "light_yellow", "light_green",
                "light_off", "light_back", "stop_sign", "speed_sign", "emergency"]

COCO_MAP = {0: "person", 1: "person", 2: "car", 3: "car", 5: "car", 7: "car", 9: "light", 11: "stop_sign"}


class Detector:
    def __init__(self, root: Path, model_path: str, fallback: str, device: str = "auto", imgsz: int = 960):
        self.ok = False
        self.custom = False
        self.names: dict[int, str] = {}
        self.error = ""
        self.imgsz = imgsz
        self.device = device
        try:
            import torch
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # noqa: BLE001
            self.error = f"YOLO not available ({e}); colour fallback only"
            return
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        custom = root / model_path
        # stock model is downloaded into models/ the first time
        path = str(custom) if custom.exists() else str(root / "models" / fallback)
        try:
            self.model = YOLO(path)
            self.custom = custom.exists()
            self.names = dict(self.model.names)
            # fp16 on the GPU is ~2x faster; only pass the flag there (newer ultralytics warns on CPU)
            self.extra = {"half": True} if self.device.startswith("cuda") else {}
            # warm-up so the first real frame isn't slow
            self.model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=imgsz, device=self.device,
                               verbose=False, **self.extra)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.error = f"could not load {path}: {e}"

    def detect(self, frame: np.ndarray, min_score: float = 0.3) -> list[Obj]:
        if not self.ok:
            return []
        res = self.model.predict(frame, imgsz=self.imgsz, conf=min_score, device=self.device,
                                 verbose=False, max_det=60, **self.extra)[0]
        out: list[Obj] = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, conf):
            name = self.names.get(int(c), str(c))
            if self.custom:
                kind, extra = self._custom_kind(name)
            else:
                kind = COCO_MAP.get(int(c))
                extra = {}
            if kind is None:
                continue
            out.append(Obj(kind=kind, box=tuple(int(v) for v in b), score=float(s), extra=extra))
        return out

    @staticmethod
    def _custom_kind(name: str) -> tuple[str | None, dict]:
        if name.startswith("light_"):
            return "light", {"state": name[6:]}
        if name in ("car", "person", "cone", "stop_sign", "speed_sign", "emergency"):
            return name, {}
        return None, {}
