"""Read the ER:LC speedometer every frame (~1 ms) by matching digit shapes.

OCR is accurate but slow on a CPU (a read can take a few hundred ms, and speeds were
smoothed over several reads), so at 200 km/h FSD was steering with a speed from a
second ago. This reader learns what each digit looks like FROM the OCR: whenever OCR
reads e.g. "127 KM/H", the three white digit shapes in front of "KM/H" are cut out,
normalised and stored as examples of 1, 2 and 7. Once it knows the digits it reads
the number itself every frame by nearest-shape matching, and OCR keeps running in
the background only to check it and to learn more.

The number is the first "word" of white text on the speedometer's top line: the
digits sit close together, then there's a gap before "KM/H".
"""
from __future__ import annotations

import cv2
import numpy as np

GW, GH = 14, 20            # normalised glyph size


def _white(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = mask.shape[0]
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if h < max(5, 0.06 * H) or a < 8 or w > 3 * h or h > 0.55 * H:
            continue                      # specks, underlines, and the box's own border
        if a < 0.12 * w * h and w * h > 400:
            continue                      # a hollow frame, not a glyph
        out.append((x, y, w, h))
    return out


def number_glyphs(img: np.ndarray) -> list[np.ndarray]:
    """The digit shapes of the speed number (left to right), normalised."""
    if img is None or img.size == 0:
        return []
    m = _white(img)
    comps = _components(m)
    if not comps:
        return []
    # the speed text is the tallest line: keep components with the most common height band
    hmax = max(c[3] for c in comps)
    line = [c for c in comps if c[3] >= 0.6 * hmax]
    if not line:
        return []
    top = min(c[1] for c in line)
    line = [c for c in line if c[1] < top + 0.5 * hmax]
    line.sort(key=lambda c: c[0])
    # first word: stop at a gap wider than ~0.3 x the typical glyph height (digits sit
    # ~4 px apart, the space before "KM/H" is ~3-4x that)
    hmed = float(np.median([c[3] for c in line]))
    word = [line[0]]
    for c in line[1:]:
        gap = c[0] - (word[-1][0] + word[-1][2])
        if gap > 0.3 * hmed:
            break
        word.append(c)
    out = []
    for x, y, w, h in word[:3]:
        g = m[y:y + h, x:x + w].astype(np.float32)
        # keep the aspect: pad to GW:GH before resizing (a "1" stays thin)
        tw = max(w, int(round(h * GW / GH)))
        pad = np.zeros((h, tw), np.float32)
        x0 = (tw - w) // 2
        pad[:, x0:x0 + w] = g
        out.append(cv2.resize(pad, (GW, GH), interpolation=cv2.INTER_AREA))
    return out


class DigitReader:
    def __init__(self, max_per_digit: int = 6):
        self.examples: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}
        self.max_per = max_per_digit
        self.reads = 0
        self.agree = 0

    @property
    def known(self) -> int:
        return sum(1 for v in self.examples.values() if v)

    def learn(self, img: np.ndarray, digits: str) -> bool:
        """OCR said the speed is `digits` (e.g. "127"): store the shapes."""
        digits = digits.strip()
        if not digits.isdigit():
            return False
        gl = number_glyphs(img)
        if len(gl) != len(digits):
            return False
        for d, g in zip(digits, gl):
            ex = self.examples[d]
            if len(ex) < self.max_per:
                ex.append(g)
            else:
                ex[self.reads % self.max_per] = g
        self.reads += 1
        return True

    def read(self, img: np.ndarray) -> tuple[int, float] | None:
        """(speed number, confidence 0..1) or None if a digit isn't known yet / unclear."""
        gl = number_glyphs(img)
        if not gl:
            return None
        labels, bank = [], []
        for d, ex in self.examples.items():
            for e in ex:
                labels.append(d)
                bank.append(e)
        if not bank:
            return None
        bank = np.stack(bank)
        val, worst = "", 1.0
        for g in gl:
            dists = np.abs(bank - g[None]).mean(axis=(1, 2))
            per_digit: dict[str, float] = {}
            for lab, dd in zip(labels, dists):
                per_digit[lab] = min(per_digit.get(lab, 1e9), float(dd))
            ranked = sorted(per_digit.items(), key=lambda kv: kv[1])
            best_d, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 1e9
            if best_d is None or best > 0.22:
                return None                            # a shape we haven't learned
            conf = 1.0 - best / 0.22
            if second < 1e9:
                conf = min(conf, (second - best) / max(second, 1e-6) * 2.0)
            worst = min(worst, conf)
            val += best_d
        if worst < 0.15:
            return None
        return int(val), float(worst)

    # ---- kept between runs: before this, every start had to re-learn the digits from the slow
    # OCR, so for the first minutes of every drive the speed was up to ~2 s old ----
    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            arrs = {f"d{d}": np.stack(ex) for d, ex in self.examples.items() if ex}
            if arrs:
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrs)
                tmp.replace(path)
        except Exception:  # noqa: BLE001
            pass

    def load(self, path) -> bool:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return False
        try:
            with np.load(path) as z:
                for d in range(10):
                    k = f"d{d}"
                    if k in z.files:
                        ex = [e.astype(np.float32) for e in z[k] if e.shape == (GH, GW)]
                        self.examples[str(d)] = ex[: self.max_per]
            return self.known > 0
        except Exception:  # noqa: BLE001
            return False

    def check(self, digits_value: int, mine: int | None) -> None:
        if mine is not None:
            self.agree += int(mine == digits_value)
