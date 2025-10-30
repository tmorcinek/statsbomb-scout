"""Tests for action_valuation module using real StatsBomb data."""

import pandas as pd
import pytest
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

from src.action_valuation import calculate_xg_value, calculate_xt_values
from src.xthreat import get_default_xt_model


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


class TestCalculateActionValues:

    def test_goal_value_is_one(self, sample_game, xt_model):
        game, events = sample_game

        goals_df = goals(events)
        print(goals_df)
        assert len(goals_df) == 3, "2:1 final score expected in sample data"

        for _, goal in goals_df.iterrows():
            assert calculate_xg_value(goal) == 1.0, "Goal action should have value 1.0"

    def test_goal_value_is_one_for_actions(self, sample_game, xt_model):
        game, events = sample_game

        goals_df = goals(events)
        actions = spadl.statsbomb.convert_to_actions(events, home_team_id=game['home_team_id'], xy_fidelity_version=2)
        print(goals_df)
        assert len(goals_df) == 3, "2:1 final score expected in sample data"

        for _, goal in goals_df.iterrows():
            assert calculate_xg_value(goal) == 1.0, "Goal action should have value 1.0"

    def test_shot_values(self, sample_game, xt_model):
        game, events = sample_game

        goals_df = goals(events)
        shots_df = shots(events).loc[~shots(events).index.isin(goals_df.index)]
        assert len(shots_df) == 13, "13 shots expected in sample data"

        for _, shot in shots_df.iterrows():
            assert calculate_xg_value(shot) != 1.0, "Non-goal shot action should not have value 1.0"
            assert calculate_xg_value(shot) > 0.0, "Shot action should be bigger than 0.0"
        assert calculate_xg_value(shots_df.iloc[0]) == pytest.approx(0.028932061, abs=1e-4), "First shot xG value mismatch"
        assert calculate_xg_value(shots_df.iloc[1]) == pytest.approx(0.07174964, abs=1e-4), "First shot xG value mismatch"
        assert calculate_xg_value(shots_df.iloc[2]) == pytest.approx(0.18899514, abs=1e-4), "First shot xG value mismatch"
        assert calculate_xg_value(shots_df.iloc[3]) == pytest.approx(0.039829366, abs=1e-4), "First shot xG value mismatch"

    def test_passes_values(self, sample_game, xt_model):
        game, events = sample_game
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        passes_df = events[events['type_name'] == 'Pass']
        first_pass = passes_df.head(1)
        actions = spadl.statsbomb.convert_to_actions(first_pass, home_team_id=game['home_team_id'], xy_fidelity_version=2)

        rates = calculate_xt_values(actions, xt_model)
        assert len(rates) == len(actions) , "xT rate shape mismatch"
        assert rates[0] == pytest.approx(0.01821798, abs=1e-4), "Pass xT value mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
