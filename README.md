# ER:LC FSD v4

A camera-based "Full Self-Driving" system for **Emergency Response: Liberty County** (Roblox). It reads the game screen, figures out the road, lights, signs and cars, and drives with W/A/S/D. It's built for the 2026 summer map, an RTX 2060 and a 1080p screen.

> **Use it on your own private server.** It presses keys for you. Roblox treats automation like that as botting, so using it on public servers can get your account banned.

## What it does

| Feature | How |
|---|---|
| Lane keeping | Turns the road into a bird's-eye view, keeps only thin paint on asphalt (sidewalks, poles and kerbs are ignored), fits curves in metres and smooths them over time. |
| Smooth steering | Pure pursuit with speed-based lookahead, road-curve feed-forward and **yaw damping** (this fixes the old "overshoot, then swerve" problem). Keys are pulsed at 60 Hz with sigma-delta, so 30% steering really averages 30%. |
| Speed | Reads the **speedometer** (kph or mph, gas or electric, finds it by itself) and **speed-limit signs**, then slows for curves. |
| Traffic lights | YOLO finds the signal heads. The lit lamp is read by its *position* in the head, because ER:LC's yellow frame fools colour counting. It ignores **signal backs** and picks the light over *your* lane. |
| Outages | A red that flashes with no green or yellow is treated as an **all-way stop**. |
| Stop signs | Stops at the **stop bar**, not at the sign. Waits a full stop, then goes once cross traffic is clear. |
| Cars, people, cones | Follows at a time gap, stops for people and cones, and keeps right for emergency lights. |
| Q / E turns | Q = left, E = right (the same keys as ER:LC's indicators). Shows a preview path, turns at the next junction, or changes lanes when the line is dashed. Press the same key again to cancel. |
| Camera | Keeps your camera aimed down the road. Gemini can also check the angle every few seconds. |
| Learning | Always keeps the last 30 s. When you take over, it asks "What went wrong?" and Gemini diagnoses the drive. The fix is **replayed on your saved drives first** and only applied if nothing gets worse. Each fix is saved as a numbered version, and F6 undoes it. |
| Training mode | Record your own driving (F2), have Gemini label it, check the labels, and train an ER:LC-specific detector on your GPU for free. |

## Setup (once)

1. Install **Python 3.11 or 3.12** from python.org and tick *Add python.exe to PATH*.
2. Download this repo (green **Code** button, then **Download ZIP**) and unzip it.
3. Double-click **`setup.bat`**. It installs everything, including GPU PyTorch (about 2.5 GB).
4. Optional but recommended: open `.env` in Notepad and paste your free keys:
   - Gemini: https://aistudio.google.com/apikey
   - Groq: https://console.groq.com/keys

## Driving

1. Run **`start.bat`**. The dashboard opens on your second monitor, or as a window.
2. In ER:LC, use the **default chase camera** behind the car.
3. **Calibrate once:** drive slowly in the middle of a straight, marked lane and press **F4**. If it says it couldn't see the lines, use `tools.bat` → 2 and click them yourself.
4. Click into Roblox and press **F1**.

| Key | Action |
|---|---|
| **F1** | FSD on / off |
| **W A S D** (or arrows, space) | Instant takeover |
| **Q / E** | Queue a left / right turn (same key again cancels) |
| **F2** | Training recorder on / off |
| **F3** | Re-aim the camera now |
| **F4** | Auto-calibrate |
| **F6** | Undo the last learned fix |
| **F7** | Open the "What went wrong?" box |
| **C** (on the dashboard) | Show / hide the camera debug view |

FSD pauses by itself if Roblox isn't the active window, so it never types into another app. If it loses the road, it stops and asks you to take over.

## Making it better (training)

1. Press **F2** and drive around the map normally for 1–2 hours: downtown, highway, rural, tunnels, night.
2. `tools.bat` → 5: Gemini labels the frames.
3. `tools.bat` → 6: check them. Press **A** to accept, **X** to reject, or click a wrong box to delete it.
4. `tools.bat` → 7: trains `models/erlc.pt` on your 2060. The app uses it automatically next start.

## Tuning by hand

Every number lives in `fsd/params.py`, with its safe range and a one-line explanation. Learned versions are saved in `data/params/` (`v0001.json` …). To change a value, add a new version or edit `current.txt` to switch versions.

## Testing without Roblox

```
python -m pytest -q tests                                     # 26 unit tests
python -m fsd.tools.inspect_image screenshot.png              # what perception sees
python -m fsd --replay folder_of_screenshots --dry-run        # whole app, no key presses
python -m fsd.learning.replay                                 # score saved disengages
```

## Layout

```
fsd/
  app.py              wires everything together (threads, hotkeys)
  capture.py          dxcam screen capture
  winput.py           SendInput keys/mouse, keyboard hook, 60 Hz actuator
  params.py           every tunable number + versioning/rollback
  settings.py         screen, calibration, keys, AI models (.env keys)
  perception/         lanes, lights, signs, YOLO, OCR, tracking, calibration
  planning/           planner (rules), turns (Q/E, lane changes)
  control/            steering + speed controllers, camera aim
  learning/           recorder, Gemini/Groq, replay tests, training
  ui/                 dashboard (pygame), debug drawing
tests/                unit tests
```

## Limits (honest)

- The first drives use the stock COCO model plus colour checks. It gets much better after training mode on your own footage.
- Turns use painted cues (stop bar, crosswalk, light, stop sign) to find the junction. Unmarked junctions need a visible cue.
- Calibration assumes Roblox's default 70° camera field of view. If you changed it, set `vfov_deg` in `data/settings.json`.
- The ER:LC 2026 map has no published road data, so routing is up to you with Q/E.
