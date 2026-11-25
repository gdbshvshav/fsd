"""Background learning loop: disengage review, Groq tuning, Gemini camera checks.

Flow after you take over:
  1. the recorder saves the last 30 s (+3 s of your correction)
  2. the dashboard asks "What went wrong?"
  3. Gemini gets ~10 frames, the telemetry and your note, and returns a
     diagnosis + proposed parameter changes
  4. the change is replayed on that scenario and the standard set
  5. passes -> saved as a new params version (old one kept for rollback)
     fails  -> nothing changes; you're told what was tried
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..notify import GREEN, ORANGE, PURPLE, RED, discord
from ..params import SPEC, ParamStore, clamp_changes, describe_spec
from .ai import CAMERA_PROMPT, Gemini, Groq
from .recorder import load_scenario, update_meta

DIAG_PROMPT = """You are the review engineer for "ER:LC FSD", a camera-based self-driving system for the
Roblox game Emergency Response: Liberty County. It reads the screen, finds lanes, traffic lights,
stop signs, cars and the speedometer, then presses W/A/S/D. The driver just DISENGAGED (took over).

Driving rules it should follow: stop at the stop line (not the sign) for red lights and stop signs;
flashing red = all-way stop; yellow = stop if comfortable else clear; follow cars at a time gap;
stop for people/cones; Q/E queue left/right turns; slow when unsure; keep right for emergency vehicles.

You get: the driver's note, telemetry sampled every 0.5 s (t, mode, reason, speed kph, target kph,
steer -1..1, pedal -1..1, light state, stop distances, lane confidence/offset, keys the HUMAN pressed),
the images (index i = the i-th image attached, times given), and the current tunable parameters.

Find what went wrong and propose a fix ONLY as changes to the tunable parameters below (stay in range,
change at most 4, prefer small changes). If the problem is perception that no parameter can fix
(e.g. a missed object), say so and leave "fix" empty.

Tunable parameters:
{spec}

Current values:
{values}

Driver's note: "{note}"
Disengage happened at t={t_dis:.1f}s.

Telemetry:
{telemetry}

Image times (s): {times}

Return JSON only:
{{"what_happened": str, "likely_cause": str, "category": "lane"|"steering"|"speed"|"light"|"stop_sign"|
"following"|"turn"|"perception"|"camera"|"other", "key_frames": [int], "fix": {{"param": value}},
"confidence": number 0..1, "advice_for_driver": str}}"""

TUNE_SYSTEM = """You tune a self-driving controller for a Roblox driving game. You get one minute of driving
statistics and the tunable parameters with ranges. Suggest at most 3 small changes (each within 15% of the
current value) that would most improve smoothness and safety, or none. JSON only:
{"changes": {"param": value}, "why": str}"""


class Learner:
    def __init__(self, store: ParamStore, settings, is_engaged, camera_aim=None):
        self.store = store
        self.settings = settings
        self.is_engaged = is_engaged
        self.camera_aim = camera_aim
        self.gemini = Gemini(settings.gemini_key, settings.gemini_model)
        self.groq = Groq(settings.groq_key, settings.groq_model)
        self.q: queue.Queue = queue.Queue()
        self.log: deque[str] = deque(maxlen=12)
        self._replayer = None
        self._stop = threading.Event()
        self.last_camera_check = 0.0
        self.busy = ""

    def start(self) -> "Learner":
        threading.Thread(target=self._run, name="learner", daemon=True).start()
        msg = []
        msg.append("Gemini on" if self.gemini.ok else "Gemini off (add GEMINI_API_KEY to .env)")
        msg.append("Groq on" if self.groq.ok else "Groq off (add GROQ_API_KEY to .env)")
        self.say(" | ".join(msg))
        return self

    def stop(self) -> None:
        self._stop.set()

    def say(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}"
        self.log.append(line)
        print("[learn]", text)
        discord().log(line)

    # ---- requests from the app ----
    def diagnose(self, scenario: Path, note: str) -> None:
        update_meta(scenario, note=note)
        self.q.put(("diagnose", scenario, note))

    def camera_check(self, frame: np.ndarray) -> None:
        now = time.time()
        if not self.gemini.ok or now - self.last_camera_check < self.settings.ai_camera_check_s:
            return
        self.last_camera_check = now
        self.q.put(("camera", frame.copy()))

    def tune(self, stats: dict) -> None:
        if self.groq.ok:
            self.q.put(("tune", stats))

    # ---- worker ----
    def _replay(self):
        if self._replayer is None:
            from .replay import Replayer
            self._replayer = Replayer(self.settings)
        return self._replayer

    def _run(self) -> None:
        deferred: list = []
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                job = None
            if job is None:
                if deferred and not self.is_engaged():
                    job = deferred.pop(0)
                else:
                    continue
            kind = job[0]
            try:
                if kind == "camera":
                    self._camera(job[1])
                elif kind == "diagnose":
                    self._diagnose(job[1], job[2], deferred)
                elif kind == "tune":
                    self._tune(job[1], deferred)
                elif kind == "local":
                    if self.is_engaged():
                        deferred.append(job)       # replays wait until you're driving yourself
                    else:
                        self._local_fix(job[1])
                elif kind == "validate":
                    if self.is_engaged():
                        deferred.append(job)       # heavy replays wait until you're driving yourself
                    else:
                        self._validate(*job[1:])
            except KeyError as e:                       # a key was rejected: say it ONCE, clearly
                which = "Gemini" if "Gemini" in str(e) else "Groq"
                url = "https://aistudio.google.com/apikey" if which == "Gemini" else "https://console.groq.com/keys"
                self.say(f"{which} key in .env is not valid - make a new one at {url}. "
                         f"Learning still works without it (local tuning).")
                discord().embed(f"{which} key rejected", f"Make a new key at {url} and put it in .env.", RED)
                if kind == "diagnose":
                    self.q.put(("local", job[1]))
            except Exception as e:  # noqa: BLE001
                self.say(f"{kind} failed: {str(e)[:120]}")
            finally:
                self.busy = ""

    def _camera(self, frame: np.ndarray) -> None:
        r = self.gemini.ask(CAMERA_PROMPT, [frame], timeout=20)
        yaw, pitch, zoom = float(r.get("yaw_deg", 0)), float(r.get("pitch_deg", 0)), int(r.get("zoom_steps", 0))
        if self.camera_aim and (abs(yaw) > 1 or abs(pitch) > 1 or zoom):
            # half-strength: the local aim loop does the fine work
            self.camera_aim.suggest(yaw * 0.5, pitch * 0.5, zoom)
            self.say(f"camera: {r.get('note', '')} (yaw {yaw:+.0f}, pitch {pitch:+.0f}, zoom {zoom:+d})")

    def _diagnose(self, d: Path, note: str, deferred: list) -> None:
        if not self.gemini.ok:
            self.say(f"saved {d.name}. Add GEMINI_API_KEY for automatic diagnosis.")
            frames, _, meta = load_scenario(d)
            t_dis = meta.get("disengage_t", 0)
            near = [f for t, f in frames if t <= t_dis][-3:]
            discord().embed(f"Disengage saved: {d.name}", "No working Gemini key, so I'm tuning from your "
                            "correction instead.", ORANGE, [("Your note", note or "(none)", False)], files=near)
            self.q.put(("local", d))
            return
        self.busy = "Gemini reviewing the disengage..."
        frames, rows, meta = load_scenario(d)
        t_dis = meta.get("disengage_t", rows[-1]["t"] if rows else 0)
        # 10 images: 8 from the 12 s before the disengage, 2 after
        before = [f for f in frames if t_dis - 12 <= f[0] <= t_dis]
        after = [f for f in frames if f[0] > t_dis]
        pick = []
        if before:
            idx = np.linspace(0, len(before) - 1, min(8, len(before))).astype(int)
            pick += [before[i] for i in idx]
        if after:
            idx = np.linspace(0, len(after) - 1, min(2, len(after))).astype(int)
            pick += [after[i] for i in idx]
        imgs = [cv2.imread(str(f)) for _, f in pick]
        times = [round(t, 1) for t, _ in pick]
        tel, last = [], -1.0
        for r in rows:
            if r["t"] - last >= 0.5 and r["t"] >= t_dis - 15:
                last = r["t"]
                tel.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                            if k in ("t", "mode", "reason", "speed", "target", "steer", "pedal", "light",
                                     "light_d", "stop_line", "stop_sign", "lane_conf", "lane_off", "keys", "engaged")})
        values = {k: round(v, 3) for k, v in self.store.snapshot().items()}
        prompt = DIAG_PROMPT.format(spec=describe_spec(), values=json.dumps(values), note=note or "(none)",
                                    t_dis=t_dis, telemetry="\n".join(json.dumps(x) for x in tel), times=times)
        res = self.gemini.ask(prompt, [i for i in imgs if i is not None], timeout=90)
        update_meta(d, diagnosis=res)
        self._post_review(d, note, res, pick, tel, t_dis)
        self.say(f"Gemini: {res.get('what_happened', '?')} | cause: {res.get('likely_cause', '?')}")
        if res.get("advice_for_driver"):
            self.say(f"tip: {res['advice_for_driver']}")
        fix = clamp_changes(res.get("fix") or {})
        if not fix:
            self.say("Gemini proposed no setting change - trying local tuning from your correction.")
            self.q.put(("local", d))
            return
        self.q.put(("validate", fix, d, "gemini", f"disengage {d.name}: {res.get('likely_cause', '')[:80]}"))

    # ------------------------------------------------------------------ local learning
    LOCAL_CANDIDATES = {
        # what you did when you took over -> which settings could explain it
        "steer": [{"steer_gain": 1.15}, {"steer_gain": 0.87}, {"steer_damping": 1.3}, {"steer_damping": 0.75},
                  {"lookahead_base_m": 1.2}, {"lookahead_base_m": 0.85}, {"lane_smoothing": 1.3}],
        "brake": [{"comfort_decel": 0.85}, {"stop_line_offset_m": "+0.5"}, {"follow_time_gap_s": 1.15},
                  {"speed_over_limit_kph": "-2"}, {"curve_lat_accel": 0.88}, {"light_to_stop_m": "+2"}],
        "throttle": [{"speed_over_limit_kph": "+2"}, {"curve_lat_accel": 1.12}, {"throttle_kp": 1.15},
                     {"follow_time_gap_s": 0.9}, {"stop_sign_wait_s": 0.8}],
    }

    def _local_fix(self, d: Path) -> None:
        """No AI needed: look at what YOU did after taking over, try the settings that
        could explain it, replay each on your saved drives, keep the best one."""
        from .replay import standard_suite, validate
        frames, rows, meta = load_scenario(d)
        t_dis = meta.get("disengage_t", 0)
        after = [r for r in rows if t_dis <= r["t"] <= t_dis + 3 and not r.get("engaged")]
        if len(after) < 5:
            self.say("Not enough of your driving after the takeover to learn from.")
            return
        steer = np.mean([(1 if "D" in r.get("keys", []) else 0) - (1 if "A" in r.get("keys", []) else 0) for r in after])
        pedal = np.mean([(1 if "W" in r.get("keys", []) else 0) - (1 if "S" in r.get("keys", []) else 0) for r in after])
        if abs(steer) >= 0.15:
            kind = "steer"
        elif pedal <= -0.15:
            kind = "brake"
        elif pedal >= 0.15:
            kind = "throttle"
        else:
            self.say("Couldn't tell what you corrected (no clear steering/braking after the takeover).")
            return
        self.busy = f"Learning from your {kind} correction..."
        cur = self.store.snapshot()
        cands = []
        for c in self.LOCAL_CANDIDATES[kind]:
            ch = {}
            for k, v in c.items():
                ch[k] = cur[k] + float(v) if isinstance(v, str) else cur[k] * v
            cands.append(clamp_changes(ch))
        if kind == "steer":                     # you kept steering one way -> shift where we sit in the lane
            cands.append(clamp_changes({"lane_center_bias_m": cur["lane_center_bias_m"] + 0.25 * np.sign(steer)}))
        rep = self._replay()
        base = rep.score(d, cur).total
        best, best_score = None, base
        for ch in cands:
            cand = dict(cur)
            cand.update(ch)
            sc = rep.score(d, cand).total
            if sc < best_score * 0.97:
                best, best_score = ch, sc
        if best is None:
            self.say(f"Tried {len(cands)} {kind} settings on this clip; none matched what you did better.")
            discord().embed("Local tuning: no change", f"You corrected **{kind}**. Tried {len(cands)} settings, "
                            "none drove closer to what you did.", ORANGE)
            return
        self.say(f"Best {kind} fix for this clip: {best} ({base:.3f} -> {best_score:.3f}); checking other drives...")
        ok, report = validate(rep, cur, best, d, standard_suite())
        self._validate_result(best, d, "local", f"you corrected {kind} in {d.name}", ok, report, cur)

    def _validate_result(self, changes, target, source, note, ok, report, before) -> None:
        diff = "\n".join(f"`{k}`: {before[k]:.3g} -> {v:.3g}" for k, v in changes.items())
        who = {"gemini": "Gemini (from a disengage)", "groq": "Groq (tuning)",
               "local": "Local learning (from your correction)"}.get(source, source)
        if ok:
            v = self.store.apply(changes, note, source)
            self.say(f"Applied {changes} as v{v} (undo: F6).")
            discord().embed(f"Setting change APPLIED (now v{v})", note, GREEN,
                            [("Changed", diff, False), ("Suggested by", who, True),
                             ("Undo", "Press F6 in FSD", True), ("Replay test", report[-1000:], False)])
            if target is not None:
                update_meta(target, fix_applied=v)
        else:
            self.say(f"Rejected {changes}: {report.splitlines()[-1]}")
            discord().embed("Setting change REJECTED (nothing changed)", note, RED,
                            [("Tried", diff, False), ("Suggested by", who, True),
                             ("Why rejected", report[-1000:], False)])
            if target is not None:
                update_meta(target, fix_rejected={"changes": changes, "report": report})

    def _post_review(self, d: Path, note: str, res: dict, pick: list, tel: list, t_dis: float) -> None:
        """Send Gemini's review to Discord with the exact frames it looked at."""
        cur = self.store.snapshot()
        fix = clamp_changes(res.get("fix") or {})
        fix_txt = "\n".join(f"`{k}`: {cur[k]:.3g} -> {v:.3g}" for k, v in fix.items()) or \
            "No setting change (perception problem or nothing to tune)"
        key = [int(i) for i in (res.get("key_frames") or []) if isinstance(i, (int, float)) and 0 <= i < len(pick)]
        order = key + [i for i in range(len(pick)) if i not in key]
        files = [pick[i][1] for i in order][:10]
        times = ", ".join(f"{pick[i][0] - t_dis:+.1f}s" for i in order[:10])
        tail = [r for r in tel if r.get("t", 0) <= t_dis][-4:]
        tel_txt = "\n".join(f"{r.get('t', 0) - t_dis:+.1f}s {r.get('mode')} {r.get('speed')}kph -> "
                            f"{r.get('target')} | {r.get('reason')} | light {r.get('light')}" for r in tail)
        discord().embed(
            f"Disengage review: {res.get('category', '?')}  ({d.name})",
            f"**What happened:** {res.get('what_happened', '?')}\n**Likely cause:** {res.get('likely_cause', '?')}",
            PURPLE,
            [("Your note", note or "(none)", False),
             ("Confidence", f"{float(res.get('confidence', 0) or 0):.0%}", True),
             ("Frames reviewed", f"{len(pick)} (key frames first): {times}", True),
             ("Proposed fix (being tested on saved drives)", fix_txt, False),
             ("Last seconds before takeover", f"```{tel_txt[:950]}```" if tel_txt else "-", False),
             ("Tip", res.get("advice_for_driver", "") or "-", False)],
            files=files, footer="Images: the frames Gemini looked at, key frames first")

    def _tune(self, stats: dict, deferred: list) -> None:
        cur = self.store.snapshot()
        spec = {k: {"value": round(cur[k], 4), "min": s.lo, "max": s.hi, "doc": s.doc} for k, s in SPEC.items()}
        res = self.groq.ask(TUNE_SYSTEM, json.dumps({"stats": stats, "params": spec}))
        ch = {}
        for k, v in clamp_changes(res.get("changes") or {}).items():
            base = cur[k]
            lim = abs(base) * 0.15 if base else (SPEC[k].hi - SPEC[k].lo) * 0.05
            ch[k] = float(np.clip(v, base - lim, base + lim))
        ch = dict(list(ch.items())[:3])
        if ch:
            self.say(f"Groq suggests {ch} - testing when you're driving manually")
            self.q.put(("validate", ch, None, "groq", f"tuning: {res.get('why', '')[:80]}"))

    def _validate(self, changes: dict, target: Path | None, source: str, note: str) -> None:
        from .replay import standard_suite, validate
        self.busy = "Testing a fix on saved drives..."
        before = self.store.snapshot()
        ok, report = validate(self._replay(), before, changes, target, standard_suite())
        self._validate_result(changes, target, source, note, ok, report, before)
        if not ok and source == "gemini" and target is not None:
            self.q.put(("local", target))             # Gemini's idea failed: learn from your correction
