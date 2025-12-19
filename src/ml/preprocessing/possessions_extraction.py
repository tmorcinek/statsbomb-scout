"""Analysis module for possessions ended with shots and goals."""

import warnings

import pandas as pd
import socceraction.spadl as spadl

warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

PITCH_LENGTH = 105
PITCH_WIDTH = 68

"""
type_id 11: shot, 12: shot_penalty, 13: shot_freekick
"""

def extract_actions_from_events(game: pd.Series, events_df: pd.DataFrame) -> pd.DataFrame:
    actions = spadl.statsbomb.convert_to_actions(events_df, game['home_team_id'], xy_fidelity_version=2)
    actions = enrich_actions_with_event_data(actions, events_df)
    return actions


def enrich_actions_with_event_data(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    cols_to_use = ['event_id', 'team_name', 'player_name', 'possession', 'type_name']

    return (
        actions
        .assign(original_event_id=lambda x: x['original_event_id'].bfill())
        .merge(
            events[cols_to_use].rename(columns={'event_id': 'original_event_id'}),
            on='original_event_id',
            how='left'
        )
    )


def normalize_pitch(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    if actions.iloc[0]['team_id'] == home_team_id:
        return actions

    # Rotate start position
    actions.loc[:, 'start_x'] = PITCH_LENGTH - actions['start_x']
    actions.loc[:, 'start_y'] = PITCH_WIDTH - actions['start_y']

    # Rotate end position
    actions.loc[:, 'end_x'] = PITCH_LENGTH - actions['end_x']
    actions.loc[:, 'end_y'] = PITCH_WIDTH - actions['end_y']

    return actions


def extract_possessions(game: pd.Series, events: pd.DataFrame) -> dict[int, pd.DataFrame]:
    actions = extract_actions_from_events(game, events)
    # actions = (spadl.add_names(actions))
    return {pid: normalize_pitch(group, game['home_team_id']) for pid, group in actions.groupby('possession')}


def extract_possessions_with_shots(game: pd.Series, events: pd.DataFrame) -> dict[int, pd.DataFrame]:
    possessions = extract_possessions(game, events)
    return {pid: df for pid, df in possessions.items() if df['type_id'].isin([11, 12, 13]).any()}


def extract_possessions_ended_with_shots(game: pd.Series, events: pd.DataFrame) -> dict[int, pd.DataFrame]:
    possessions = extract_possessions(game, events)
    return {pid: df for pid, df in possessions.items() if df.iloc[-1]['type_id'] in [11, 12, 13]}


def extract_possessions_ended_with_goals(game: pd.Series, events: pd.DataFrame) -> dict[int, pd.DataFrame]:
    possessions = extract_possessions(game, events)
    return {pid: df for pid, df in possessions.items() if df.iloc[-1]['type_id'] in [11, 12, 13] and df.iloc[-1]['result_id'] == 1}
