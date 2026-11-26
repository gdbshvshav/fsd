"""What the driving core did, kept in memory for the dashboard and for logs.

* A rolling 60 s buffer of the important signals (lane error, steering, pedal,
  speed, target speed, yaw raw/fused, lane confidence).
* Simple quality numbers the dashboard shows and the Discord log reports:
  lateral error (mean/max), steering reversals per second (twitchiness),
  how often the raw yaw disagreed with the fused one, lane dropouts.
* dump_csv(): write the buffer to data/logs/drive_*.csv (called on disengage, so a
  bad moment can be looked at afterwards).
"""
from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

import numpy as np

FIELDS = ("t", "e", "psi", "steer", "pedal", "v", "v_target", "yaw_raw", "yaw", "lane_conf",
          "lane_raw_off", "lane_off", "mode")


class Telemetry:
    def __init__(self, seconds: float = 60.0):
        self.rows: deque = deque(maxlen=int(seconds * 60))
        self._sense = {}

    def add_sense(self, t, raw_lanes, lanes, yaw_raw, yaw, v_raw, v) -> None:
        self._sense = {"lane_raw_off": raw_lanes.offset if raw_lanes is not None else 0.0,
                       "lane_off": lanes.offset, "lane_conf": lanes.conf, "yaw_raw": yaw_raw, "yaw": yaw, "v": v}

    def add_act(self, t, plan, steer, pedal, dbg) -> None:
        r = dict(self._sense)
        r.update({"t": t, "e": dbg.get("e", 0.0), "psi": dbg.get("psi", 0.0), "steer": steer, "pedal": pedal,
                  "v_target": dbg.get("v_target", 0.0), "mode": plan.mode.value})
        self.rows.append(r)

    def quality(self, window_s: float = 20.0) -> dict:
        if not self.rows:
            return {}
        t1 = self.rows[-1]["t"]
        rs = [r for r in self.rows if t1 - r["t"] <= window_s]
        if len(rs) < 10:
            return {}
        e = np.array([r["e"] for r in rs])
        s = np.array([r["steer"] for r in rs])
        ds = np.sign(np.diff(s))
        ds = ds[ds != 0]
        rev = int(np.sum(ds[1:] != ds[:-1])) if len(ds) > 1 else 0
        span = max(1e-3, rs[-1]["t"] - rs[0]["t"])
        conf = np.array([r.get("lane_conf", 0.0) for r in rs])
        return {"lat_err_mean": float(np.mean(np.abs(e))), "lat_err_max": float(np.max(np.abs(e))),
                "steer_reversals_per_s": rev / span, "lane_dropout": float(np.mean(conf < 0.2))}

    def dump_csv(self, folder: str | Path = "data/logs", tag: str = "drive") -> Path | None:
        if not self.rows:
            return None
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            wr.writeheader()
            for r in self.rows:
                wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        return path
