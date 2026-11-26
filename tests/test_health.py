"""Start-up health check wording."""
from fsd.health import check
from fsd.world import Lanes, World


def test_health_flags_slow_pc_and_unread_speed():
    w = World(speed_valid=False)
    w.lanes = Lanes(conf=0.8, offset=0.1)
    rep = check({"lanes_fps": 8}, w, 18.0, 30.0)
    assert not rep.ok
    assert any("fps" in p for p in rep.problems) and any("speed" in p for p in rep.problems)


def test_health_all_good():
    w = World(speed_valid=True, speed_kph=60)
    w.lanes = Lanes(conf=0.8, offset=0.2)
    rep = check({"lanes_fps": 30, "night_gain": 2.0, "road_kind": "town"}, w, 58.0, 30.0)
    assert rep.ok and any("night" in line for line in rep.lines)
