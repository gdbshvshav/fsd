"""Train an ER:LC-specific detector from your own drives (free, on your GPU).

1. Drive with F2 on: frames are saved to data/train/raw (every 0.5 s while moving).
2. python -m fsd.learning.training label    -> Gemini draws boxes on each frame
3. python -m fsd.learning.training review   -> you check the boxes (keys shown on screen)
4. python -m fsd.learning.training train    -> trains YOLO on your RTX card, saves models/erlc.pt

The app uses models/erlc.pt automatically next time it starts. It knows ER:LC's
light heads (and their yellow backs), cones, stop signs, speed signs and
emergency lights, which the stock model doesn't.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2

from ..perception.detector import ERLC_CLASSES
from ..settings import Settings

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "train" / "raw"
LAB = ROOT / "data" / "train" / "labels_raw"      # Gemini output (json per image)
DS = ROOT / "data" / "train" / "dataset"           # YOLO format

LABEL_PROMPT = """Detect objects for a self-driving system in this screenshot of the Roblox game ER:LC.
Labels (use exactly these):
- car: any vehicle (not the player's own car at the bottom centre, ignore that one)
- person: pedestrians / players on foot
- cone: traffic cones, barrels, road barriers
- light_red / light_yellow / light_green: a traffic signal head FACING the camera, by its lit lamp
- light_off: a signal head facing the camera with no lamp lit
- light_back: the back or side of a signal head (often a plain yellow box)
- stop_sign: red octagon facing the camera
- speed_sign: speed limit sign facing the camera
- emergency: police/fire/ambulance vehicle with red/blue lights on
Return JSON only: {"objects": [{"label": str, "box_2d": [ymin, xmin, ymax, xmax]}]} with coordinates
normalised to 0-1000. Include small distant traffic lights. Return {"objects": []} if none."""


def label(limit: int = 0) -> None:
    from .ai import Gemini
    s = Settings.load()
    g = Gemini(s.gemini_key, s.gemini_model, per_minute=10)
    if not g.ok:
        raise SystemExit("Add GEMINI_API_KEY to .env first.")
    LAB.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(RAW.glob("*.jpg")) if not (LAB / (p.stem + ".json")).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} images to label")
    for i, p in enumerate(todo):
        img = cv2.imread(str(p))
        if img is None:
            continue
        for attempt in range(3):
            try:
                res = g.ask(LABEL_PROMPT, [img], timeout=60)
                break
            except RuntimeError as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    time.sleep(15)
                    continue
                print("  error:", e)
                res = None
                break
        else:
            res = None
        if res is None:
            continue
        objs = [o for o in res.get("objects", []) if o.get("label") in ERLC_CLASSES and len(o.get("box_2d", [])) == 4]
        (LAB / (p.stem + ".json")).write_text(json.dumps({"objects": objs, "reviewed": False}))
        print(f"  [{i + 1}/{len(todo)}] {p.name}: {len(objs)} objects")


def review() -> None:
    """OpenCV reviewer. Keys: A accept, X reject image, click a box to delete it, ESC quit."""
    files = sorted(LAB.glob("*.json"))
    files = [f for f in files if not json.loads(f.read_text()).get("reviewed")]
    print(f"{len(files)} to review. A = accept, X = throw away, click box = delete it, Esc = stop")
    cols = {"car": (255, 170, 60), "person": (60, 200, 255), "cone": (0, 140, 255), "light_red": (40, 40, 240),
            "light_yellow": (0, 210, 255), "light_green": (60, 220, 60), "light_off": (150, 150, 150),
            "light_back": (90, 90, 90), "stop_sign": (40, 40, 230), "speed_sign": (230, 230, 230),
            "emergency": (255, 60, 255)}
    state = {"objs": [], "scale": 1.0, "W": 1, "H": 1}

    def on_click(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = x / state["scale"], y / state["scale"]
        for o in list(state["objs"]):
            y1, x1, y2, x2 = [v / 1000 for v in o["box_2d"]]
            if x1 * state["W"] <= x <= x2 * state["W"] and y1 * state["H"] <= y <= y2 * state["H"]:
                state["objs"].remove(o)
                break

    cv2.namedWindow("review")
    cv2.setMouseCallback("review", on_click)
    for f in files:
        img = cv2.imread(str(RAW / (f.stem + ".jpg")))
        if img is None:
            continue
        data = json.loads(f.read_text())
        state["objs"] = data["objects"]
        state["H"], state["W"] = img.shape[:2]
        state["scale"] = min(1.0, 1400 / state["W"])
        while True:
            view = img.copy()
            for o in state["objs"]:
                y1, x1, y2, x2 = [int(v / 1000 * d) for v, d in zip(o["box_2d"], [state["H"], state["W"]] * 2)]
                c = cols.get(o["label"], (255, 255, 255))
                cv2.rectangle(view, (x1, y1), (x2, y2), c, 2)
                cv2.putText(view, o["label"], (x1, max(15, y1 - 4)), 0, 0.6, c, 2)
            view = cv2.resize(view, None, fx=state["scale"], fy=state["scale"])
            cv2.putText(view, f"{f.stem}  A=accept X=reject click=delete box Esc=quit", (10, 25), 0, 0.7, (0, 255, 255), 2)
            cv2.imshow("review", view)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("a"), ord("A")):
                data["objects"], data["reviewed"] = state["objs"], True
                f.write_text(json.dumps(data))
                break
            if k in (ord("x"), ord("X")):
                data["reviewed"], data["rejected"] = True, True
                f.write_text(json.dumps(data))
                break
            if k == 27:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


def build_dataset(include_unreviewed: bool = False) -> Path:
    if DS.exists():
        shutil.rmtree(DS)
    for split in ("train", "val"):
        (DS / "images" / split).mkdir(parents=True)
        (DS / "labels" / split).mkdir(parents=True)
    items = []
    for f in sorted(LAB.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("rejected") or (not d.get("reviewed") and not include_unreviewed):
            continue
        items.append((f, d))
    random.Random(0).shuffle(items)
    n_val = max(1, len(items) // 10)
    for i, (f, d) in enumerate(items):
        split = "val" if i < n_val else "train"
        src = RAW / (f.stem + ".jpg")
        if not src.exists():
            continue
        shutil.copy(src, DS / "images" / split / src.name)
        lines = []
        for o in d["objects"]:
            y1, x1, y2, x2 = [v / 1000 for v in o["box_2d"]]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)
            if w > 0 and h > 0:
                lines.append(f"{ERLC_CLASSES.index(o['label'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (DS / "labels" / split / (f.stem + ".txt")).write_text("\n".join(lines))
    yaml = DS / "erlc.yaml"
    yaml.write_text(f"path: {DS.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
                    "".join(f"  {i}: {n}\n" for i, n in enumerate(ERLC_CLASSES)))
    print(f"dataset: {len(items) - n_val} train / {n_val} val images")
    return yaml


def train(epochs: int = 80, base: str = "yolo11s.pt", include_unreviewed: bool = False) -> None:
    from ultralytics import YOLO
    yaml = build_dataset(include_unreviewed)
    model = YOLO(str(ROOT / "models" / base))
    # RTX 2060 (6 GB): batch 8 at 960 px fits
    model.train(data=str(yaml), epochs=epochs, imgsz=960, batch=8, patience=20,
                project=str(ROOT / "data" / "train" / "runs"), name="erlc", exist_ok=True, verbose=True)
    best = ROOT / "data" / "train" / "runs" / "erlc" / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / "erlc.pt"
        if dst.exists():
            shutil.copy(dst, ROOT / "models" / f"erlc_backup_{time.strftime('%Y%m%d_%H%M')}.pt")
        shutil.copy(best, dst)
        print(f"saved {dst} - restart the app to use it")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["label", "review", "train", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--unreviewed", action="store_true", help="also train on labels you haven't checked")
    a = ap.parse_args()
    if a.step in ("label", "all"):
        label(a.limit)
    if a.step in ("review", "all"):
        review()
    if a.step in ("train", "all"):
        train(a.epochs, include_unreviewed=a.unreviewed)


if __name__ == "__main__":
    main()
