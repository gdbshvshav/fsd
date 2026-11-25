"""Replay saved scenarios through perception + planner and score the driving.

Used to TEST every proposed fix before it goes live:
  1. the scenario that caused the disengage must get better, and
  2. the standard set of saved scenarios must not get worse.

Scoring (lower = better):
  * imitation: after you took over, how close our steering/pedal would have
    been to what you actually did
  * violations: frames where we'd keep going through a red light / stop sign
  * jerkiness: how much the steering command jumps around

    python -m fsd.learning.replay                     # score all scenarios with current params
    python -m fsd.learning.replay data/scenarios/X    # one scenario, verbose
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..control.controllers import LateralController, SpeedController
from ..params import ParamStore, SPEC, defaults
from ..planning.planner import Planner
from ..settings import Settings
from ..world import Light, Mode
from .recorder import list_scenarios, load_scenario

PERCEPTION_KEYS = ("lane_smoothing", "lane_width_m", "light_min_score", "object_min_score", "flash_window_s")


@dataclass
class Score:
    imitation: float = 0.0
    violations: int = 0
    jerk: float = 0.0
    frames: int = 0
    notes: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.imitation + 0.15 * self.violations + 2.0 * self.jerk


def _human_cmd(rows: list[dict], t: float, win: float = 0.3) -> tuple[float, float] | None:
    sel = [r for r in rows if abs(r["t"] - t) <= win and not r.get("engaged", False)]
    if not sel:
        return None
    steer = np.mean([(1 if "D" in r.get("keys", []) else 0) - (1 if "A" in r.get("keys", []) else 0) for r in sel])
    pedal = np.mean([(1 if "W" in r.get("keys", []) else 0) - (1 if "S" in r.get("keys", []) else 0) for r in sel])
    return float(steer), float(pedal)


class Replayer:
    def __init__(self, settings: Settings | None = None, use_detector: bool = True):
        from ..fast.pipeline import FastPerception
        self.settings = settings or Settings.load()
        self.use_detector = use_detector
        self.perception = FastPerception(self.settings, use_detector=use_detector, use_ocr=False, live=False)
        self.cache: dict[tuple, list] = {}

    def _worlds(self, d: Path, p: dict):
        key = (str(d),) + tuple(round(p[k], 4) for k in PERCEPTION_KEYS)
        if key in self.cache:
            return self.cache[key]
        frames, rows, meta = load_scenario(d)
        per = self.perception
        per.geo = None                                    # fresh trackers per scenario
        per.tracker = type(per.tracker)()
        per.ego_shape = type(per.ego_shape)()
        worlds = []
        for t, f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            w = per.process(img, p, t=t)
            row = min(rows, key=lambda r: abs(r["t"] - t)) if rows else {}
            w.speed_kph = float(row.get("speed", 0.0))
            w.speed_valid = bool(row.get("speed_valid", True))
            if row.get("limit"):
                w.speed_limit_kph = row["limit"]
            worlds.append((t, w))
        self.cache[key] = (worlds, rows, meta)
        if len(self.cache) > 64:
            self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def score(self, d: Path, p: dict, verbose: bool = False) -> Score:
        worlds, rows, meta = self._worlds(d, p)
        planner, lat, lon = Planner(), LateralController(), SpeedController()
        sc = Score()
        prev_steer = None
        errs = []
        t_dis = meta.get("disengage_t", 1e9)
        last_t = None
        for t, w0 in worlds:
            w = copy.copy(w0)
            planner.last_t = last_t if last_t is not None else t - 0.1
            plan = planner.step(w, p, engaged=True, now=t)
            dt = 0.1 if last_t is None else max(0.02, t - last_t)
            last_t = t
            steer = lat.step(w, plan, p, dt)
            pedal = lon.step(w, plan, p, dt)
            sc.frames += 1
            if prev_steer is not None:
                sc.jerk += abs(steer - prev_steer)
            prev_steer = steer
            # rule checks on what perception saw
            near_stop = plan.stop_at is not None and plan.stop_at < 3
            red = w.signal.state in (Light.RED, Light.FLASHING_RED) or w.stop_sign_dist < 6
            if red and w.speed_kph > 8 and near_stop and plan.target_speed_kph > 8 and plan.mode != Mode.TURNING:
                sc.violations += 1
            if t >= t_dis:
                h = _human_cmd(rows, t)
                if h is not None:
                    errs.append(abs(steer - h[0]) + 0.5 * abs(np.sign(pedal) - np.sign(h[1])) * min(1, abs(h[1]) * 2))
            if verbose:
                print(f"t={t:6.2f} {plan.mode.value:11s} v={w.speed_kph:5.1f} tgt={plan.target_speed_kph:5.1f} "
                      f"steer={steer:+.2f} pedal={pedal:+.2f} light={w.signal.state.value} "
                      f"lane={w.lanes.conf:.2f} | {plan.reason}")
        sc.imitation = float(np.mean(errs)) if errs else 0.0
        sc.jerk = sc.jerk / max(1, sc.frames)
        return sc


def validate(replayer: Replayer, base: dict, changes: dict, target: Path | None,
             suite: list[Path]) -> tuple[bool, str]:
    """Accept only if the target scenario improves and nothing else gets worse."""
    cand = dict(base)
    cand.update(changes)
    lines = []
    if target is not None:
        b = replayer.score(target, base)
        c = replayer.score(target, cand)
        lines.append(f"target {target.name}: {b.total:.3f} -> {c.total:.3f}")
        if not (c.total < b.total * 0.97 or (c.violations < b.violations)):
            return False, "\n".join(lines + ["no improvement on the scenario that failed"])
    worse = 0
    sum_b = sum_c = 0.0
    for d in suite:
        if target is not None and d == target:
            continue
        b = replayer.score(d, base)
        c = replayer.score(d, cand)
        sum_b += b.total
        sum_c += c.total
        if c.violations > b.violations or c.total > b.total * 1.05 + 0.01:
            worse += 1
            lines.append(f"  worse: {d.name} {b.total:.3f} -> {c.total:.3f} (violations {b.violations}->{c.violations})")
    if worse:
        return False, "\n".join(lines + [f"{worse} saved scenario(s) got worse"])
    if target is None:
        # a tuning tweak with no specific failure must make the suite better overall
        if not suite or sum_c > sum_b * 0.98:
            return False, "\n".join(lines + [f"suite {sum_b:.3f} -> {sum_c:.3f}: not enough improvement"])
    lines.append(f"passed: {len(suite)} scenario(s) checked, none worse")
    return True, "\n".join(lines)


def standard_suite(max_n: int = 12) -> list[Path]:
    """Tagged 'standard' scenarios first (red light, stop sign, Q/E turns, tunnel, highway exit), then newest."""
    import json
    allp = list_scenarios()
    tagged = [d for d in allp if "standard" in json.loads((d / "meta.json").read_text()).get("tags", [])]
    rest = [d for d in reversed(allp) if d not in tagged]
    return (tagged + rest)[:max_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?")
    ap.add_argument("--no-detector", action="store_true")
    a = ap.parse_args()
    store = ParamStore()
    p = store.snapshot()
    r = Replayer(use_detector=not a.no_detector)
    targets = [Path(a.scenario)] if a.scenario else list_scenarios()
    if not targets:
        print("No scenarios yet. They're saved automatically when you take over from FSD.")
    for d in targets:
        s = r.score(d, p, verbose=bool(a.scenario))
        print(f"{d.name}: total {s.total:.3f}  imitation {s.imitation:.3f}  violations {s.violations}  "
              f"jerk {s.jerk:.3f}  ({s.frames} frames)")


if __name__ == "__main__":
    main()

__all__ = ["Replayer", "validate", "standard_suite", "Score", "SPEC", "defaults"]
