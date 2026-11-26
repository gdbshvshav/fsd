"""Drive FSD through the REAL Liberty County map (ER:LC), not hand-made test roads.

The map comes from erlc-tools.com's 3D map export: every part of the game map
(position, rotation, size, colour, material) plus the terrain height field. It
is downloaded once on your PC and cached in data/erlc_map/ (it is NOT stored in
this repository - it's their work).

    mapdata.py   load / download / cache the map, ground height, spatial lookups
    glrender.py  OpenGL renderer of the map from ER:LC's chase camera (moderngl)
    roads.py     lane lines and road surfaces pulled out of the parts -> routes
    signals.py   finds the traffic-light lamps so they can be switched red/yellow/green
    drive.py     closed-loop drives on real routes, scored like fsd.sim
"""
