"""Automatic tuning of the driving parameters in the simulator.

    python -m fsd.sim tune                     # quick scenarios, 3 rounds
    python -m fsd.sim tune --rounds 5 --names highway_curves truck_curves

How it works (coordinate search, no AI):

1. Score the current parameters on the chosen scenarios (lower = better; a failed
   rule like a red light or a collision costs a lot, lane error and comfort a little).
2. For each tunable parameter in turn, try a step up and a step down (inside the
   parameter's safe bounds from params.py). Keep a change only if the total score
   gets better.
3. Halve the step sizes and repeat for the next round.

Scenarios run in parallel processes (one per CPU core). The best parameters are
written to data/sim_tuned.json and can be loaded as a new parameter version from
the dashboard or with `python -m fsd.sim run all --params data/sim_tuned.json`.

Only the controller / planner numbers are tuned here; perception thresholds are
left alone because the simulator's picture is cleaner than the real game's.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..params import SPEC, defaults

# parameters worth tuning in the simulator and the starting step (fraction of their range)
TUNABLE: dict[str, float] = {
    "steer_gain": 0.15,
    "steer_ff_gain": 0.15,
    "steer_damping": 0.15,
    "lookahead_base_m": 0.12,
    "lookahead_gain_s": 0.12,
    "steer_rate_limit": 0.12,
    "curve_lat_accel": 0.12,
    "comfort_decel": 0.10,
    "follow_time_gap_s": 0.08,
    "stop_line_offset_m": 0.10,
}

OUT = Path(__file__).resolve().parents[2] / "data" / "sim_tuned.json"


def _run_one(args) -> tuple[str, float, bool]:
    name, params = args
    from .runner import SimRun
    from .scenarios import SCENARIOS
    m = SimRun(SCENARIOS[name], params, fine=False).run()
    return name, m.score(), m.passed()


class Evaluator:
    def __init__(self, names: list[str], workers: int | None = None):
        self.names = names
        self.workers = workers or max(1, min(len(names), (os.cpu_count() or 2) - 1))
        self.cache: dict[str, tuple[float, int]] = {}
        self.calls = 0

    def score(self, params: dict) -> tuple[float, int]:
        key = json.dumps({k: round(v, 5) for k, v in sorted(params.items())})
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        jobs = [(n, params) for n in self.names]
        if self.workers > 1:
            with ProcessPoolExecutor(self.workers) as ex:
                res = list(ex.map(_run_one, jobs))
        else:
            res = [_run_one(j) for j in jobs]
        total = sum(r[1] for r in res)
        passed = sum(1 for r in res if r[2])
        self.cache[key] = (total, passed)
        return total, passed


def tune(names: list[str], rounds: int = 3, start: dict | None = None, out: Path = OUT,
         log=print) -> dict:
    p = dict(defaults())
    if start:
        p.update(start)
    ev = Evaluator(names)
    t0 = time.time()
    best, passed = ev.score(p)
    log(f"start: score {best:.2f}, {passed}/{len(names)} passed  ({', '.join(names)})")
    steps = {k: frac * (SPEC[k].hi - SPEC[k].lo) for k, frac in TUNABLE.items() if k in SPEC}
    for rnd in range(1, rounds + 1):
        improved = False
        for k, step in steps.items():
            spec = SPEC[k]
            for sign in (+1, -1):
                trial = dict(p)
                trial[k] = min(spec.hi, max(spec.lo, p[k] + sign * step))
                if abs(trial[k] - p[k]) < 1e-9:
                    continue
                sc, ok = ev.score(trial)
                if sc < best - 1e-6:
                    log(f"  round {rnd}: {k} {p[k]:.3f} -> {trial[k]:.3f}   score {best:.2f} -> {sc:.2f} "
                        f"({ok}/{len(names)} passed)")
                    best, passed, p = sc, ok, trial
                    improved = True
                    break
        steps = {k: v / 2 for k, v in steps.items()}
        if not improved:
            log(f"  round {rnd}: no better values found")
    changed = {k: round(p[k], 4) for k in TUNABLE if k in p and abs(p[k] - defaults()[k]) > 1e-9}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(changed, indent=1))
    log(f"done in {time.time() - t0:.0f}s, {ev.calls} evaluations. best score {best:.2f}, "
        f"{passed}/{len(names)} passed. changes -> {out}")
    for k, v in changed.items():
        log(f"  {k}: {defaults()[k]} -> {v}")
    return p
