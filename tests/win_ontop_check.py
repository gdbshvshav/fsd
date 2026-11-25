"""Windows-only check: FSD dashboard must stay visible on top while another
full-screen window (like Roblox) has focus. Run by GitHub Actions on Windows."""
import ctypes
import subprocess
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

u = ctypes.windll.user32
root_dir = Path(__file__).resolve().parent.parent
img_dir = root_dir / "data" / "ontop_test"
img_dir.mkdir(parents=True, exist_ok=True)
img = np.full((1080, 1920, 3), (70, 70, 70), np.uint8)
cv2.line(img, (700, 1080), (930, 600), (235, 235, 235), 8)
cv2.line(img, (1220, 1080), (990, 600), (235, 235, 235), 8)
cv2.imwrite(str(img_dir / "road.png"), img)

proc = subprocess.Popen([sys.executable, "-m", "fsd", "--replay", str(img_dir), "--dry-run"], cwd=root_dir)
hwnd = 0
for _ in range(120):
    hwnd = u.FindWindowW(None, "ER:LC FSD")
    if hwnd:
        break
    time.sleep(0.5)
assert hwnd, "FSD window never appeared"
print("FSD window found", hwnd)
time.sleep(3)

# Fake "Roblox": a borderless full-screen window that grabs focus
fake = tk.Tk()
fake.title("Roblox")
fake.configure(bg="#335")
fake.attributes("-fullscreen", True)
fake.update()
fake.focus_force()
fake.lift()
fake_hwnd = int(fake.wm_frame(), 16)
u.SetForegroundWindow(fake_hwnd)

results = []
t0 = time.time()
while time.time() - t0 < 12:
    fake.update()
    time.sleep(0.2)
    if time.time() - t0 < 4:
        continue
    r = wintypes.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(r))
    cx, cy = (r.left + r.right) // 2, (r.top + r.bottom) // 2
    top = u.GetAncestor(u.WindowFromPoint(wintypes.POINT(cx, cy)), 2)
    ex = u.GetWindowLongW(hwnd, -20)
    results.append(dict(alive=proc.poll() is None, visible=bool(u.IsWindowVisible(hwnd)),
                        minimized=bool(u.IsIconic(hwnd)), topmost=bool(ex & 0x8), on_top_at_center=(top == hwnd),
                        foreground_is_fake=(u.GetForegroundWindow() == fake_hwnd)))

fake.destroy()
proc.terminate()
last = results[-1]
print("last check:", last)
bad = [r for r in results if not (r["alive"] and r["visible"] and not r["minimized"] and r["on_top_at_center"])]
print(f"{len(results) - len(bad)}/{len(results)} checks passed")
if bad:
    print("first failure:", bad[0])
    sys.exit(1)
print("PASS: FSD stays visible on top of a focused full-screen window")
