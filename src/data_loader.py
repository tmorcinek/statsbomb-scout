"""Module for loading StatsBomb event data."""

from typing import Generator

import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader


def load_statsbomb_data(loader: StatsBombLoader, competition_id: int, season_id: int) -> Generator[pd.DataFrame, None, None]:
    games = loader.games(competition_id, season_id)
    print(f"Found {len(games)} matches in competition {competition_id}, season {season_id}")

    for index, game in games.iterrows():
        game_id = game['game_id']
        print(f"Loading events for match {game_id} ({index + 1}/{len(games)})...")

        try:
            yield loader.events(game_id)
        except Exception as e:
            print(f"Warning: Failed to load events for match {game_id}: {e}")
            continue
