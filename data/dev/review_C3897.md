# Review of detections on C3897.MP4 (5:18, second sample video)

No full labels: the annotator answered yes/no for each detection of the first run (68 events).
The fixes below were derived from these answers; counts are from re-running on the same video.

| Class | Confirmed / reported (first run) | Finding | Fix | After |
|---|---|---|---|---|
| illegal_u_turn | 1 / 7 | cars entering/leaving at the left, right or top border: the cut-off box slides and fakes a turn | ignore samples at the side/top border | 1 (confirmed) |
| near_miss | – / 4 | tiny cars at the far end of the avenue; cars cut off at the border | conflicts need ≥ 4.5 % of frame width, ignore border samples | 1 (unanswered) |
| accident | – / 1 | same | same | 0 |
| stopped_vehicle | 5 / 7 | the two rejected ones were kerbside cars on the far carriageway | must be overtaken by traffic in its own direction; far carriageway excluded | 5 (all confirmed) |
| failure_to_yield | 5 / 11 | rejected ones had the pedestrian 1.9–2.2 car lengths away (further along the crossing, not in the path); confirmed ones 0.5–1.4 | pedestrian within 1.5 car lengths | 7 (4 confirmed, 3 rejected) |
| jaywalking | 6 / 8 | one rejected "pedestrian" moved at ~2 body heights/s through the lanes: a rider on an undetected scooter/bike | walking speed 0.4–1.6 body heights/s | 16 |
| congestion | 0 / 2 (1 unanswered) | 22 s episodes = busy red phases | direction at a standstill ≥ 30 s | 1 (unanswered) |
| stop_line | 1 / 2 (1 unanswered) | – | – | 3 |

Answers as given: A (stopped_vehicle) y y n y y y n · B (failure_to_yield) n n n y y y n n y y n ·
C (jaywalking) y y y n n y y y · congestion n n · stop_line y n · U-turn at 0:43: yes.
