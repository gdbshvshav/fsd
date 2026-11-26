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
| **M** (on the dashboard) | Show / hide the mini map (bottom right) |
| **V** (on the dashboard) | Switch camera view / 3D road view |

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
python -m pytest -q tests                                     # unit + simulator tests
python -m fsd.tools.inspect_image screenshot.png              # what perception sees
python -m fsd --replay folder_of_screenshots --dry-run        # whole app, no key presses
python -m fsd.learning.replay                                 # score saved disengages
```

## Driving simulator (closed loop)

FSD can drive a small ER:LC-style world on its own, so every change can be checked
before it reaches the game. The simulator draws the game picture (same chase camera,
70° FOV, road colours, blue shadows, yellow signal heads, HUD speedometer), feeds it
through the *exact* perception / planning / control code the app uses, turns the
controller's output into W/A/S/D presses through the same key modulator, and moves a
car with keyboard-style Roblox physics (steering value ramps while A/D is held and
springs back when released).

```
python -m fsd.sim list                        # the scenarios
python -m fsd.sim run all                     # drive them all and score
python -m fsd.sim run highway_curves --video out   # also write an mp4 of what FSD saw
python -m fsd.sim run quick --coarse          # faster, lower-resolution picture
python -m fsd.sim tune --rounds 3             # search better controller numbers
```

Each run reports lane error, lane departures, wheels over the edge/median, red lights
run, stop signs rolled, collisions, closest gap and comfort, and says PASS/FAIL.

Scenarios: straight and curved highway, yellow-edge highway, shadows, recovering from an
offset start, red light, light cycling while approaching, stop sign, power outage
(flashing red), lead car braking hard, parked car in the lane, heavy truck on bends,
tight town bend, and starting with the wrong calibration.

## Driving the REAL Liberty County map

The simulator can also drive the actual ER:LC map: every part, mesh and the terrain
from erlc-tools.com's 3D map export, drawn from the chase camera with OpenGL
(your graphics card; any GPU works). The map is downloaded once into `data/erlc_map/`
(it isn't stored in this repo - it's their work). Scoring uses the map's own lane paint.

```
.venv\Scripts\pip install -r requirements-sim.txt     # once: moderngl + meshoptimizer
python -m fsd.sim.erlc list
python -m fsd.sim.erlc run all
python -m fsd.sim.erlc run town_south_north --video out --verbose
python -m fsd.sim.erlc shot highway_east --out view.png
```

## Driving on its own for hours

- **Turn signals**: ER:LC's blinkers are Q (left) and E (right). Whenever FSD changes
  lanes, merges or turns, it taps Q/E to put the blinker on, and taps again to switch it
  off afterwards. If *you* put a blinker on, FSD leaves it for you to cancel.
- **Lane ends / merges**: when our lane's lines converge ahead (2 lanes becoming 1, an
  on-ramp joining the highway), FSD signals and merges into the lane that continues,
  waiting for a gap.
- **Road ends**: at a T-junction or dead end (junction ahead, no road straight on) it
  picks the way that has road - right if both - signals and turns, stopping first if
  there's a stop sign or a red light.
- Turn these off in the settings with `auto_merge = 0` / `auto_route = 0`.

## How it drives (short version)

The driving core lives in `fsd/drive/` (param `drive_core` = 1; set it to 0 for the old one):

- **Speed** is the speedometer reading *plus* what the pedals should be doing in between
  readings, so it's never stale (a misread "70" for "20" is ignored unless it repeats).
- **Yaw** (how fast we turn) comes from the steering we hold and the lanes; the picture
  is only trusted when it agrees. (The old picture-only yaw had spikes that steered the
  car off the road at 100 km/h.)
- **Lanes** are Kalman-filtered at 60 Hz: smooth, predicted between detections, re-based
  the moment we cross a line, coasted straight across junctions.
- **Steering**: an LQR controller on the path (rebuilt for each speed) plus the bend's
  curvature ahead, then converted to the Roblox steering value. Roblox keyboard steering
  isn't an angle - holding A/D moves a hidden steering value at a fixed rate and
  releasing springs it back. One frame of A/D is too coarse for motorway speed, so small
  steering is **dithered** (sigma-delta): on average the car gets exactly the steering
  wanted - no dead band, no drifting at 200 km/h.
- **Speed planning** looks 200 m ahead: bends (from the path and the lane shape), stop
  points and the limit; brakes early and smoothly (jerk-limited), never late.
- **Which side**: ER:LC drives on the right. A yellow line on our right or a car coming at
  us in our lane = wrong side -> signal and move right. No lane lines -> the road's edges
  are found from the road colour and we keep to the right half (or the middle of a narrow
  road). On fast multi-lane roads we return to the right lane after passing.
- **Supervisor**: if the car drifts while steering against it, the steering sits at full
  lock, lanes are unreliable, the speed can't be read or the PC is slow, it eases the
  speed down and tells you why.
- **Junctions**: stop bars are measured while visible and *remembered* (dead-reckoned
  from our speed) after they slide under our own car; a stop sign or flashing red seen
  for the junction means a full stop at the bar, kept until the stop is done
  (`planning/junction_memory.py`). A red light that leaves the picture while we're
  stopped is held until it's seen to change.
- **Cars ahead**: anything in our lane that isn't road colour is an obstacle (no AI
  needed). A close car disappears behind our own car with the chase camera, so the
  lead car is remembered and predicted (`planning/lead_memory.py`) - a far-away misread
  of its roof can't make us drive into it.
- **Traffic lights**: lit lamps found by brightness + colour inside a housing; a car's
  two tail lights side by side are recognised and ignored.

## Layout

```
fsd/
  app.py              wires everything together (threads, hotkeys)
  capture.py          dxcam screen capture
  winput.py           SendInput keys/mouse, keyboard hook, 60 Hz actuator
  params.py           every tunable number + versioning/rollback
  settings.py         screen, calibration, keys, AI models (.env keys)
  perception/         lanes, lights, signs, YOLO, OCR, tracking, calibration
  fast/               the 60 fps vision path: paint, lanes, lamps, obstacles, pipeline
  planning/           planner (rules), turns (Q/E, lane changes), junction + lead memory
  drive/              driving core: yaw/speed/lane filters, LQR steering, speed planner,
                      pedal map, right-hand-traffic rules, supervisor, telemetry
  control/            Roblox steering model (key dithering), old controllers, camera aim
  sim/                closed-loop driving simulator, scenarios, auto-tuner
  learning/           recorder, Gemini/Groq, replay tests, training
  ui/                 dashboard (pygame), animated HUD, mini map, animation toolkit
tests/                unit tests
```

## Limits (honest)

- The first drives use the stock COCO model plus colour checks. It gets much better after training mode on your own footage.
- Turns use painted cues (stop bar, crosswalk, light, stop sign) to find the junction. Unmarked junctions need a visible cue.
- Calibration assumes Roblox's default 70° camera field of view. If you changed it, set `vfov_deg` in `data/settings.json`.
- The ER:LC 2026 map has no published road data, so routing is up to you with Q/E.
