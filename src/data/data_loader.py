"""Module for loading StatsBomb event data."""

from typing import Generator, Tuple

import pandas as pd

from statsbombpy import sb
from socceraction.data.statsbomb import StatsBombLoader


def load_statsbomb_data(competition_id: int, season_id: int) -> Generator[Tuple[pd.Series, pd.DataFrame], None, None]:
    matches = sb.matches(competition_id, season_id)
    print(f"Found {len(matches)} matches in competition {competition_id}, season {season_id}")

    for index, match in matches.iterrows():
        match_id = match['match_id']
        print(f"Loading events for match {match_id} ({index + 1}/{len(matches)})...")

        try:
            events = sb.events(match_id)
            events = _filter_relevant_events(events)
            # For compatibility with socceraction
            events['possession_team_name'] = events['possession_team']
            events['team_name'] = events['team']
            events['type_name'] = events['type']
            match["home_team_id"] = get_home_team_id(match, events)
            match["game_id"] = match['match_id']

            yield match, events
        except Exception as e:
            print(f"Warning: Failed to load events for match {match_id}: {e}")
            continue


def get_home_team_id(match: pd.Series, events: pd.DataFrame) -> int:
    team_match = events[events['team'] == match["home_team"]]

    if team_match.empty:
        raise ValueError(f"Home team '{match['home_team']}' not found in events")

    return int(team_match['team_id'].iloc[0])

def _filter_relevant_events(events) -> pd.DataFrame:
    return events[~events['type'].isin([
        'Substitution', 'Injury Stoppage', 'Half Start', 'Half End',
        'Tactical Shift', 'Referee Ball-Drop', 'Starting XI'
    ])].sort_values(by=['index'])


def load_statsbomb_socceraction_data(root: str, competition_id: int, season_id: int) -> Generator[Tuple[pd.Series, pd.DataFrame], None, None]:
    loader = StatsBombLoader(root=root, getter="local")
    games = loader.games(competition_id, season_id)
    print(f"Found {len(games)} matches in competition {competition_id}, season {season_id}")

    for index, game in games.iterrows():
        game_id = game['game_id']
        print(f"Loading events for match {game_id} ({index + 1}/{len(games)})...")

        try:
            yield game, (loader.events(game_id, load_360=True))
        except Exception as e:
            print(f"Warning: Failed to load events for match {game_id}: {e}")
            continue


def load_socceraction_data(loader: StatsBombLoader, competition_id: int, season_id: int) -> Generator[Tuple[pd.Series, pd.DataFrame], None, None]:
    games = loader.games(competition_id, season_id)
    print(f"Found {len(games)} matches in competition {competition_id}, season {season_id}")

    for index, game in games.iterrows():
        game_id = game['game_id']
        print(f"Loading events for match {game_id} ({index + 1}/{len(games)})...")

        try:
            yield game, (loader.events(game_id, load_360=True))
        except Exception as e:
            print(f"Warning: Failed to load events for match {game_id}: {e}")
            continue
