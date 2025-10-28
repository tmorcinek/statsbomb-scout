"""Module for loading StatsBomb event data."""

from typing import Generator, Tuple

import pandas as pd

from statsbombpy import sb


def load_statsbomb_data(competition_id: int, season_id: int) -> Generator[Tuple[int, pd.DataFrame], None, None]:
    matches = sb.matches(competition_id, season_id)
    print(f"Found {len(matches)} matches in competition {competition_id}, season {season_id}")

    for index, game in matches.iterrows():
        match_id = game['match_id']
        print(f"Loading events for match {match_id} ({index + 1}/{len(matches)})...")

        try:
            events = sb.events(match_id)
            events = filter_relevant_events(events)
            yield match_id, events
        except Exception as e:
            print(f"Warning: Failed to load events for match {match_id}: {e}")
            continue


def filter_relevant_events(events) -> pd.DataFrame:
    return events[~events['type'].isin([
        'Substitution', 'Injury Stoppage', 'Half Start', 'Half End',
        'Tactical Shift', 'Referee Ball-Drop', 'Starting XI'
    ])].sort_values(by=['index'])
