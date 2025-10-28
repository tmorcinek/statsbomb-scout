import multiprocessing
import os

BASE_PATH = "data/statsbomb/data/"

OPEN_DATA_PATHS = {
    "competitions": "competitions.json",
    "matches": "matches/{competition_id}/{season_id}.json",
    "lineups": "lineups/{match_id}.json",
    "events": "events/{match_id}.json",
    "frames": "three-sixty/{match_id}.json",
}

if "SB_CORES" in os.environ:
    MAX_CONCURRENCY = int(os.environ["SB_CORES"])
else:
    try:
        MAX_CONCURRENCY = max(multiprocessing.cpu_count() - 2, 4)
    except NotImplementedError:
        MAX_CONCURRENCY = 4

VERSIONS = {
    "competitions": "v4",
    "matches": "v6",
    "lineups": "v4",
    "events": "v8",
    "360-frames": "v2",
    "player-match-stats": "v5",
    "player-season-stats": "v4",
    "team-season-stats": "v2",
    "team-match-stats": "v1",
}
