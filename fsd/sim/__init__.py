"""ER:LC-style driving simulator for testing FSD without Roblox.

Why: every change to the driving brain can be tried on many roads, at speed,
in seconds - lane keeping on curves, stopping at red lights and stop signs,
following a car, shadows, a car parked in the lane - and scored, before it ever
touches your car in the real game.

It renders pictures that look like ER:LC through the same high chase camera
(concrete highways and dark asphalt, blue Roblox shadows, yellow-framed
signals, your car in the middle of the screen), feeds them to the real FSD
vision + planner + controllers, turns the commands into W/A/S/D key presses
exactly like the real actuator does, and moves the car with a Roblox-style
keyboard car model.

    python -m fsd.sim list                    # scenarios
    python -m fsd.sim run highway_curves      # run one, prints a score
    python -m fsd.sim run all --video out/    # all, with videos
    python -m fsd.sim tune                    # search controller settings that score best

Modules:
    road.py      roads as centre lines with lanes, markings, junctions
    world.py     signals, stop signs, other cars, shadows, props; a whole scene
    camera.py    chase camera that lags behind the car like Roblox's
    render.py    draws the scene into an image (painter's algorithm, clipping)
    vehicle.py   keyboard car physics (steering ramps, throttle, brakes, drag)
    scenarios.py ready-made test drives
    runner.py    closed loop: picture -> FSD -> keys -> car, with scoring
    tune.py      automatic tuning of controller settings on the scenarios
"""
