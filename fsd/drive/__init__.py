"""The driving core: state estimation, lane tracking, path and speed planning, and the
steering / pedal controllers that turn a plan into W/A/S/D.

    yaw.py            how fast the car is really turning (chassis model + picture + lanes)
    vehicle_model.py  Roblox car model: steering value -> curvature, pedal -> acceleration
    lane_filter.py    smooth, gap-free lane estimate at 60 Hz (Kalman filter)
    side_keeper.py    right-hand traffic: which part of the road is ours
    path_planner.py   the line to drive: lane centre, lane changes, keep-right, nudges
    speed_planner.py  how fast, looking ahead: bends, stops, cars; smooth (jerk-limited)
    lateral.py        steering: preview LQR + curvature feed-forward, weave guard
    longitudinal.py   pedals: acceleration control with a learned pedal map
    driver.py         puts it all together for the app and the simulator
    telemetry.py      what happened, for logs and the dashboard
"""
