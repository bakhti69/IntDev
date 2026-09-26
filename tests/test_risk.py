"""Part B score on hand-made road users: a collision course alarms, passing traffic does not."""
import numpy as np

from trafficwatch.risk import CausalRisk


class _User:
    def __init__(self, pos, vel, t, size=80.0):
        self.category, self.t, self.pos, self.size = "car", [t - 0.4, t - 0.2, t], [np.array(pos, float)], [size]
        self._vel = np.array(vel, float)

    def velocity(self):
        return self._vel

    def decel(self):
        return 0.0


def _raw(a, b):
    risk = CausalRisk.__new__(CausalRisk)
    risk.scene = type("S", (), {"wrong_way_zones": {}})()
    return risk._raw_risk([_User(*a, t=1.0), _User(*b, t=1.0)], 1.0)


def test_crossing_paths_alarm_before_impact():
    # both 1.2 car lengths from the conflict point at 2 lengths/s: impact in 0.6 s
    assert _raw(([404, 500], [160, 0]), ([500, 404], [0, 160])) > 0.5


def test_overlapping_boxes_are_occlusion_not_alarm():
    # one car drives in front of a queued one: ground points already coincide
    assert _raw(([495, 500], [200, 0]), ([500, 505], [0, 0])) < 0.2


def test_passing_a_standing_car_in_the_next_lane():
    assert _raw(([300, 500], [200, 0]), ([500, 520], [0, 0])) < 0.5
