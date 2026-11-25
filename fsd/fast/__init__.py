"""Fast, GPU-free perception for 60 fps driving.

Everything in this package is plain image processing (OpenCV + NumPy), tuned on
real ER:LC screenshots, and designed to run in a few milliseconds per frame on
an ordinary CPU:

    scaler.py        cheap 4K -> 960/1920 frame shrinking
    paint.py         road paint (lane lines, stop bars, crosswalks) in the game picture
    screen_lanes.py  lane lines found directly on screen, linked toward the vanishing point
    autocal.py       keeps the camera calibration right while you drive (zoom, FOV, car)
    lamps.py         traffic lights from their lit lamps (no AI)
    obstacles.py     cars/objects in your path from "this isn't road" (no AI)
    pipeline.py      puts it all together into a World every frame
"""
