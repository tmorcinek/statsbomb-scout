"""Module for loading StatsBomb event data."""

import pandas as pd
import json
from pathlib import Path
from typing import Union, List


class StatsBombDataLoader:
    """Loads and parses StatsBomb event data from CSV or JSON files."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader.

        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = Path(data_path)
        self.events_df = None

    def load_from_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load events from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            DataFrame with event data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TODO: Parse StatsBomb JSON structure
        # Extract relevant fields: timestamp, location, type, player, etc.
        self.events_df = pd.json_normalize(data)
        return self.events_df

    def load_from_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load events from a CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with event data
        """
        self.events_df = pd.read_csv(file_path)
        return self.events_df

    def load_multiple_matches(self, match_ids: List[int]) -> pd.DataFrame:
        """
        Load data from multiple matches.

        Args:
            match_ids: List of match IDs to load

        Returns:
            Combined DataFrame with all matches
        """
        all_events = []

        for match_id in match_ids:
            # TODO: Implement loading logic for multiple matches
            file_path = self.data_path / f"match_{match_id}.json"
            if file_path.exists():
                events = self.load_from_json(file_path)
                events['match_id'] = match_id
                all_events.append(events)

        self.events_df = pd.concat(all_events, ignore_index=True)
        return self.events_df

    def filter_relevant_events(self) -> pd.DataFrame:
        """
        Filter only relevant event types (passes, carries, receptions, shots).

        Returns:
            Filtered DataFrame
        """
        # TODO: Filter event types
        # Example: self.events_df[self.events_df['type'].isin(['Pass', 'Carry', 'Ball Receipt', 'Shot'])]
        relevant_types = ['Pass', 'Carry', 'Ball Receipt*', 'Shot']
        self.events_df = self.events_df[self.events_df['type'].isin(relevant_types)]
        return self.events_df

