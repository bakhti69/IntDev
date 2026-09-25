"""Each rule on a synthetic scenario (positive) plus a normal-traffic scene (negative)."""
from __future__ import annotations

import numpy as np

from conftest import concat, make_analysis, make_track, path, still
from trafficwatch.rules import detect


def labels(events):
    return sorted({e[2] for e in events})


def only(events, label):
    return [e for e in events if e[2] == label]


def normal_traffic():
    """Free flow in both directions, legal pedestrians: must produce no events."""
    tracks = [
        # towards the camera through the junction, following its lane
        make_track(1, "car", *path(0, 6, (205, 215), (1250, 830))),
        # away from the camera on the upper carriageway
        make_track(2, "car", *path(2, 8, (1500, 380), (420, 180))),
        # pedestrian on the main crosswalk
        make_track(3, "person", *path(10, 22, (380, 528), (980, 468))),
    ]
    return tracks


def test_normal_traffic_is_quiet():
    assert detect(make_analysis(normal_traffic())) == []


def test_wrong_way():
    tr = make_track(1, "car", *path(3, 8, (700, 360), (200, 170)))
    ev = only(detect(make_analysis([tr])), "wrong_way")
    assert len(ev) == 1 and ev[0][0] <= 3.5 and ev[0][1] >= 7.5


def test_u_turn():
    ang = np.linspace(-np.pi / 2, np.pi / 2, 76)
    t = 4 + np.arange(len(ang)) / 12.5
    pts = np.stack([1100 + 130 * np.cos(ang), 650 + 130 * np.sin(ang)], axis=1)
    lead_t, lead = path(2, 4 - 0.08, (900, 520), (1100, 520))
    tr = make_track(1, "car", *concat((lead_t, lead), (t, pts)))
    ev = only(detect(make_analysis([tr])), "illegal_u_turn")
    assert len(ev) == 1 and 3.5 <= ev[0][0] <= 5.5 and ev[0][1] >= 8.0


def test_stopped_vehicle_with_traffic_passing():
    tracks = [make_track(1, "car", *still(5, 25, (1200, 700)))]
    for k in range(5):  # traffic keeps flowing past it
        t0 = 6 + 3.5 * k
        tracks.append(make_track(10 + k, "car", *path(t0, t0 + 3, (900, 600), (1500, 820))))
    ev = only(detect(make_analysis(tracks)), "stopped_vehicle")
    assert len(ev) == 1 and abs(ev[0][0] - 5) < 0.5 and abs(ev[0][1] - 25) < 0.5


def test_queue_is_not_stopped_vehicle():
    tracks = [make_track(k, "car", *still(0, 30, (300 + 110 * k, 420 - 12 * k))) for k in range(4)]
    assert only(detect(make_analysis(tracks)), "stopped_vehicle") == []


def test_queue_inference_is_not_used_for_red_light():
    """The approach towards the camera has its signal facing away: on the sample video,
    queue-based red inference fired on turn lanes, so red_light is only read from heads."""
    tracks = [
        make_track(1, "car", *still(0, 30, (390, 415))),
        make_track(2, "car", *still(0, 30, (590, 392))),
        make_track(3, "car", *path(10, 14, (330, 300), (900, 640))),
    ]
    assert only(detect(make_analysis(tracks)), "red_light") == []


def test_stop_line_inferred_from_queue():
    """A car waits past the stop line while the queue next to it holds behind it."""
    tracks = [
        make_track(1, "car", *still(0, 30, (390, 415))),
        make_track(2, "car", *still(0, 30, (300, 425))),
        make_track(3, "car", *concat(path(2, 4, (560, 330), (640, 440)), still(4.08, 20, (640, 440)))),
    ]
    ev = only(detect(make_analysis(tracks)), "stop_line")
    assert len(ev) == 1 and 3.5 <= ev[0][0] <= 5.0 and ev[0][1] >= 19


def test_green_start_is_not_red_light():
    """Queue heads pulling away together with the crossing car: that is a green start."""
    tracks = [
        make_track(1, "car", *concat(still(0, 11, (390, 415)), path(11.08, 15, (390, 415), (800, 700)))),
        make_track(2, "car", *concat(still(0, 11, (590, 392)), path(11.08, 15, (590, 392), (1000, 650)))),
        make_track(3, "car", *path(10, 14, (330, 300), (900, 640))),
    ]
    assert only(detect(make_analysis(tracks)), "red_light") == []


def test_jaywalking():
    tr = make_track(1, "person", *path(3, 9, (700, 580), (950, 720)))
    ev = only(detect(make_analysis([tr])), "jaywalking")
    assert len(ev) == 1 and ev[0][0] < 3.5


def test_failure_to_yield():
    tracks = [
        make_track(1, "person", *path(0, 12, (430, 522), (760, 490))),
        make_track(2, "car", *path(4, 7, (420, 330), (760, 690))),
    ]
    ev = only(detect(make_analysis(tracks)), "failure_to_yield")
    assert len(ev) == 1


def test_accident():
    a = make_track(1, "car", *concat(path(3, 5, (520, 700), (930, 700)), still(5.08, 14, (930, 700))))
    b = make_track(2, "car", *concat(path(3, 5, (975, 900), (975, 720)), still(5.08, 14, (975, 720))))
    ev = only(detect(make_analysis([a, b])), "accident")
    assert len(ev) == 1 and 4.0 <= ev[0][0] <= 5.5 and ev[0][1] - ev[0][0] < 5


def test_near_miss_hard_braking():
    brake = lambda u: 1 - (1 - u) ** 3  # noqa: E731  (decelerates to a stop)
    a = make_track(1, "car", *concat(path(3, 4.5, (500, 700), (800, 700)),
                                     path(4.58, 5.2, (806, 700), (845, 700), ease=brake),
                                     still(5.28, 9, (845, 700))))
    b = make_track(2, "car", *path(3.5, 7, (960, 520), (960, 900)))
    ev = only(detect(make_analysis([a, b])), "near_miss")
    assert len(ev) == 1 and ev[0][1] - ev[0][0] < 6
    assert only(detect(make_analysis([a, b])), "accident") == []


def test_congestion_upper():
    tracks = []
    for k in range(7):
        x0 = 500 + 120 * k
        y0 = 200 + 0.37 * (x0 - 500) + (k % 2) * 25
        tracks.append(make_track(k + 1, "car", *path(0, 40, (x0, y0), (x0 - 20, y0 - 8))))
    ev = only(detect(make_analysis(tracks, duration=45)), "congestion")
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 30


def test_segments_never_overlap_within_class():
    tracks = normal_traffic() + [make_track(9, "person", *path(3, 9, (700, 580), (950, 720))),
                                 make_track(10, "person", *path(5, 12, (720, 600), (980, 740)))]
    ev = only(detect(make_analysis(tracks)), "jaywalking")
    assert len(ev) == 1  # two concurrent jaywalkers are one segment
