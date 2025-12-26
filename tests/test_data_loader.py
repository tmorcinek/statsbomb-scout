import pandas as pd
import pytest
from socceraction.data.statsbomb import StatsBombLoader

from src.analysis.game_utils import game_summary
from src.data.data_loader import _add_team_names_to_game, load_statsbomb_socceraction_data, load_socceraction_data

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@pytest.fixture(scope="module")
def statsbomb_loader():
    return StatsBombLoader(root="data/statsbomb/data", getter="local")


@pytest.fixture(scope="module")
def first_game(statsbomb_loader):
    game = statsbomb_loader.games(competition_id=55, season_id=282).iloc[0]
    return game


@pytest.fixture(scope="module")
def first_teams(statsbomb_loader, first_game):
    teams = statsbomb_loader.teams(first_game['game_id'])
    return teams


def test_first_game(statsbomb_loader):
    first_game = statsbomb_loader.games(competition_id=55, season_id=282).iloc[0]

    assert isinstance(first_game, pd.Series), "First game should be a pandas Series"

    assert first_game['season_id'] == 282
    assert first_game['competition_id'] == 55
    assert first_game['home_team_id'] == 941
    assert first_game['away_team_id'] == 768
    assert first_game['home_score'] == 1
    assert first_game['away_score'] == 2


def test_first_teams(statsbomb_loader, first_game):
    first_teams = statsbomb_loader.teams(first_game['game_id'])
    assert isinstance(first_teams, pd.DataFrame), "Teams should be a Pandas DataFrame"
    assert len(first_teams) == 2, "There should be exactly two teams in a match"

    team_1 = first_teams.iloc[0]
    team_2 = first_teams.iloc[1]

    assert team_1['team_id'] == 941
    assert team_1['team_name'] == 'Netherlands'
    assert team_2['team_id'] == 768
    assert team_2['team_name'] == 'England'


def test_first_events(statsbomb_loader, first_game):
    events = statsbomb_loader.events(first_game['game_id'])

    assert isinstance(events, pd.DataFrame), "Teams should be a Pandas DataFrame"
    assert len(events) == 3485, "Number of events does not match!"
    assert events[events['type_name'] == "Shot"].shape[0] == 16, "Number of shots does not match!"


def test_enriched_game_fixture(first_game, first_teams):
    _add_team_names_to_game(first_game, first_teams)

    assert first_game['home_team_name'] == 'Netherlands'
    assert first_game['away_team_name'] == 'England'

    assert first_game['season_id'] == 282
    assert first_game['competition_id'] == 55
    assert first_game['home_team_id'] == 941
    assert first_game['away_team_id'] == 768
    assert first_game['home_score'] == 1
    assert first_game['away_score'] == 2


def test_load_statsbomb_socceraction_data():
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))

    assert game['home_team_name'] == 'Netherlands'
    assert game['away_team_name'] == 'England'

    assert game['season_id'] == 282
    assert game['competition_id'] == 55
    assert game['home_team_id'] == 941
    assert game['away_team_id'] == 768
    assert game['home_score'] == 1
    assert game['away_score'] == 2

    assert len(events) == 3485, "Number of events does not match!"


def test_load_socceraction_data(statsbomb_loader):
    game, events = next(load_socceraction_data(statsbomb_loader, 55, 282))

    assert game['home_team_name'] == 'Netherlands'
    assert game['away_team_name'] == 'England'

    assert game['season_id'] == 282
    assert game['competition_id'] == 55
    assert game['home_team_id'] == 941
    assert game['away_team_id'] == 768
    assert game['home_score'] == 1
    assert game['away_score'] == 2

    assert len(events) == 3485, "Number of events does not match!"


def test_game_summary():
    game = pd.Series({
        'home_team_name': 'France',
        'away_team_name': 'Spain',
        'home_score': 2,
        'away_score': 1,
        'competition_stage': 'Final',
        'game_date': pd.Timestamp('2024-07-14 20:00:00')
    })

    result = game_summary(game)
    expected = 'France 2 : 1 Spain (Final) [14.07.2024]'

    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_game_summary_with_first_game(first_game, first_teams):
    _add_team_names_to_game(first_game, first_teams)

    result = game_summary(first_game)
    expected = 'Netherlands 1 : 2 England (Semi-finals) [10.07.2024]'

    assert result == expected, f"Expected '{expected}', got '{result}'"
