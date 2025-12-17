"""Tests for action_valuation module using real StatsBomb data."""

import numpy as np
import pandas as pd
import pytest
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

from src.ml.action_valuation import calculate_xg_values, calculate_xt_values
from src.ml.xthreat import get_default_xt_model

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@pytest.fixture(scope="module")
def statsbomb_loader():
    return StatsBombLoader(root="data/statsbomb/data", getter="local")


@pytest.fixture(scope="module")
def sample_game(statsbomb_loader):
    game = statsbomb_loader.games(competition_id=55, season_id=282).iloc[0]
    events = statsbomb_loader.events(game['game_id'])
    return game, events


@pytest.fixture(scope="module")
def xt_model():
    return get_default_xt_model()


def shots(events: pd.DataFrame):
    return events[events['type_name'] == 'Shot']


def goals(events: pd.DataFrame):
    return events[
        (events['type_name'] == 'Shot') &
        (events['extra'].apply(
            lambda x: isinstance(x, dict) and
                      x.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
        ))
        ]


def specific_possession(events: pd.DataFrame, possession_number: int = 11):
    possession_events = events[events['possession'] == possession_number]
    possession_events = possession_events[possession_events['team_name'] == possession_events['possession_team_name']]
    return possession_events


class TestCalculateXGValues:

    def test_goal_value_is_one(self, sample_game, xt_model):
        game, events = sample_game

        goals_df = goals(events)

        assert len(goals_df) == 3, "2:1 final score expected in sample data"
        assert (calculate_xg_values(goals_df) == 1.0).all(), "Goal action should have value 1.0"

    def test_shot_values(self, sample_game, xt_model):
        game, events = sample_game

        goals_df = goals(events)
        shots_df = shots(events)

        shots_df = shots_df.loc[~shots_df.index.isin(goals_df.index)]
        assert len(shots_df) == 13, "13 shots expected in sample data"

        shots_xg_values = calculate_xg_values(shots_df)
        assert (shots_xg_values != 1.0).all(), "Non-goal shot action should not have value 1.0"
        assert (shots_xg_values > 0.0).all(), "Shot action should be bigger than 0.0"

        fist_4_shots = shots_df.head(4)
        shots_xg_values = list(calculate_xg_values(fist_4_shots))
        assert np.allclose(shots_xg_values, [0.028932061, 0.07174964, 0.18899514, 0.039829366], atol=1e-4), "Shots xG values mismatch"

    def test_passes_values(self, sample_game, xt_model):
        game, events = sample_game

        passes_df = events[events['type_name'] == 'Pass']
        first_passes = passes_df.head(5)
        values = calculate_xg_values(first_passes)
        print(values)
        assert (values == 0.0).all(), "Pass action should have value 0.0"
        # for _, event_pass in first_passes.iterrows():
        #     assert calculate_xg_value(event_pass) == 0.0, "Pass action should have value 0.0"


class TestCalculateXTValues:

    def test_xavi_simons_goal_value(self, sample_game, xt_model):
        game, events = sample_game

        possession_with_goal = specific_possession(events, possession_number=11)

        actions = spadl.statsbomb.convert_to_actions(possession_with_goal, home_team_id=game['home_team_id'], xy_fidelity_version=2)

        rates = calculate_xt_values(actions, xt_model)

        assert sum(rates) == 0.00814035, "Goal from Xavi Simons xt value mismatch"

    def test_fake_action(self, xt_model):
        data = {
            # 'game_id': [3942819],
            # 'original_event_id': ['9448d9e0-19ae-4d91-a6fb-1fc12a0c11ec'],
            # 'period_id': [1],
            # 'time_seconds': [371.288],
            # 'team_id': [941],
            # 'player_id': [37274.0],
            'start_x': [8.00625],
            'start_y': [24.6925],
            'end_x': [104.95625],
            'end_y': [36.2525],
            'type_id': [0],  # Pass
            'result_id': [1],  # Complete
            # 'bodypart_id': [2],
            # 'action_id': [0]
        }
        actions = pd.DataFrame(data)

        rates = calculate_xt_values(actions, xt_model)

        assert sum(rates) == 0.24857372, "Pass from goalkeeper to forward on the edge of the box xt value mismatch"

    def test_hary_kane_goal_value(self, sample_game, xt_model):
        game, events = sample_game

        possession_with_goal = specific_possession(events, possession_number=19)

        actions = spadl.statsbomb.convert_to_actions(possession_with_goal, home_team_id=game['home_team_id'], xy_fidelity_version=2)

        rates = calculate_xt_values(actions, xt_model)

        assert sum(rates) == 0.0, "Penalty shot xt value should be 0.0"

    def test_passes_values(self, sample_game, xt_model):
        game, events = sample_game

        passes_df = events[events['type_name'] == 'Pass']
        first_pass = passes_df.head(1)
        id_ = game['home_team_id']
        print(id_)
        actions = spadl.statsbomb.convert_to_actions(first_pass, home_team_id=id_, xy_fidelity_version=2)

        rates = calculate_xt_values(actions, xt_model)
        assert len(rates) == len(actions), "xT rate shape mismatch"
        assert rates[0] == pytest.approx(0.01821798, abs=1e-4), "Pass xT value mismatch"



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
