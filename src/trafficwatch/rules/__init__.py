"""Event rules: each maps an :class:`~trafficwatch.pipeline.Analysis` to
``[[start_sec, end_sec, label], ...]`` for one class."""
from __future__ import annotations

import logging
from typing import Callable

from ..pipeline import Analysis
from ..segments import finalize_events
from .conflicts import accident, candidate_pairs, near_miss
from .junction import congestion, red_light, stop_line
from .pedestrians import failure_to_yield, jaywalking
from .scene_events import fire_smoke, road_obstacle
from .vehicles import illegal_turn, illegal_u_turn, solid_line_crossing, stopped_vehicle, wrong_way

log = logging.getLogger(__name__)

RULES: dict[str, Callable[[Analysis], list]] = {
    "red_light": red_light,
    "wrong_way": wrong_way,
    "illegal_u_turn": illegal_u_turn,
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "illegal_turn": illegal_turn,
    "solid_line_crossing": solid_line_crossing,
    "stop_line": stop_line,
    "congestion": congestion,
    "road_obstacle": road_obstacle,
    "fire_smoke": fire_smoke,
}

# post-processing per class: bridge gaps shorter than MERGE_GAP, drop segments shorter than MIN_LEN
MERGE_GAP = {"jaywalking": 1.5, "red_light": 2.0, "congestion": 5.0, "stopped_vehicle": 2.0, "accident": 2.0,
             "road_obstacle": 3.0, "fire_smoke": 3.0}
MIN_LEN = {"stopped_vehicle": 10.0, "congestion": 15.0, "jaywalking": 1.0, "red_light": 0.5,
           "accident": 0.5, "near_miss": 0.5, "failure_to_yield": 0.3}


def detect(an: Analysis, classes: list[str] | None = None) -> list[list]:
    """Run every rule (a failing rule is logged and skipped) and clean the segments."""
    wanted = set(classes) if classes else None
    events: list[list] = []
    pairs = None
    for label, rule in [("accident", accident), ("near_miss", near_miss), *RULES.items()]:
        if wanted is not None and label not in wanted:
            continue
        try:
            if label in ("accident", "near_miss"):
                pairs = candidate_pairs(an) if pairs is None else pairs
                events += rule(an, pairs)
            else:
                events += rule(an)
        except Exception:  # one broken rule must not cost the whole video
            log.exception("rule %s failed on %s", label, an.video_path)
    return finalize_events(events, an.info.duration, MERGE_GAP, MIN_LEN)
