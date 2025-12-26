import warnings

import pytest
import socceraction.spadl as spadl
from matplotlib import pyplot as plt

import config
from src.analysis.actions_possessions import possessions_from_actions
from src.analysis.game_utils import game_summary
from src.analysis.visualization import plot_possession_actions, plot_multiple_possessions
from src.ml.preprocessing.possessions_extraction import extract_actions_from_events, extract_possessions, extract_possessions_with_shots, \
    extract_possessions_ended_with_goals, extract_possessions_ended_with_shots

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_loader import load_socceraction_data, load_socceraction_match


@pytest.fixture(scope="module")
def sample_game():
    return load_socceraction_match("data/statsbomb/data", 55, 282)


def test_first_match_possessions(sample_game):
    game, events = sample_game

    actions_df = extract_actions_from_events(game, events)

    possessions_df = possessions_from_actions(actions_df)
    # print(f"Events:\n{possessions_df}")

    assert len(possessions_df) == 111

    at_least_6_actions_in_possession = (possessions_df['length'] >= 6).sum()
    assert at_least_6_actions_in_possession == 82

    at_least_8_actions_in_possession = (possessions_df['length'] >= 8).sum()
    assert at_least_8_actions_in_possession == 71

    at_least_10_actions_in_possession = (possessions_df['length'] >= 10).sum()
    assert at_least_10_actions_in_possession == 66


def test_first_match_possessions_spadl():
    from socceraction.data.statsbomb import StatsBombLoader
    SBL = StatsBombLoader(root="data/statsbomb/data", getter="local")
    game, events = next(load_socceraction_data(SBL, 55, 282))

    df_actions = spadl.statsbomb.convert_to_actions(events, game['home_team_id'], xy_fidelity_version=2)

    df_actions = (
        spadl
        .add_names(df_actions)  # add actiontype and result names
        .merge(SBL.teams(game_id=3942819))  # add team names
        .merge(SBL.players(game_id=3942819))  # add player names
    )
    df_actions = df_actions.drop('nickname', axis=1)
    # actions = enrich_actions_with_event_data(actions, events)

    print(f"Event columns: {len(df_actions)}")
    print(f"Total events: \n{df_actions.head(80)}\n")


def test_first_possession_england(sample_game):
    game, events = sample_game

    possessions = extract_possessions(game, events)

    england_first_possession = possessions[2]

    first_action = england_first_possession.iloc[0]

    assert first_action['team_name'] == 'England'

    assert first_action['start_x'] == 52.456250000000004  # placeholder
    assert first_action['start_y'] == 34.0425
    assert first_action['end_x'] == 22.443749999999994
    assert first_action['end_y'] == 38.7175

    plot_possession_actions(england_first_possession)
    plt.show()


def test_possession_netherlands(sample_game):
    game, events = sample_game

    possessions = extract_possessions(game, events)

    england_first_possession = possessions[11]

    first_action = england_first_possession.iloc[0]

    assert first_action['team_name'] == 'Netherlands'

    assert first_action['start_x'] == pytest.approx(8.0062, abs=1e-4)
    assert first_action['start_y'] == pytest.approx(24.6925, abs=1e-4)
    assert first_action['end_x'] == pytest.approx(18.4187, abs=1e-4)
    assert first_action['end_y'] == pytest.approx(39.1425, abs=1e-4)

    plot_possession_actions(england_first_possession)
    plt.show()


def test_possession_with_custom_title(sample_game):
    game, events = sample_game

    possessions = extract_possessions(game, events)

    possession = possessions[2]

    fig = plot_possession_actions(possession, title="Custom Possession Title - Test")
    fig.savefig('data/test/possession_2_custom_title.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_multiple_possessions_best(sample_game):
    game, events = sample_game

    possessions = extract_possessions(game, events)
    assert len(possessions) == 111

    possessions_list = [possessions[pid] for pid in sorted(possessions.keys()) if len(possessions[pid]) >= config.SEQUENCE_LENGTH]
    best_possessions = [78, 8, 68, 18, 13]
    possessions_list = [possessions_list[pid] for pid in best_possessions]

    fig = plot_multiple_possessions(possessions_list, main_title=f"{game_summary(game)}\nPossessions with ids: {best_possessions}")
    fig.savefig('data/test/visualization_best.png', dpi=300, bbox_inches='tight')

    plt.show()


def test_plot_multiple_shots(sample_game):
    game, events = sample_game

    possessions = extract_possessions_with_shots(game, events)

    assert len(possessions) == 15

    fig = plot_multiple_possessions(possessions.values(), main_title=f"{game_summary(game)}\nPossessions with Shots")
    fig.savefig('data/test/visualization_shots.png', dpi=300, bbox_inches='tight')
    # fig.canvas.manager.set_window_title("window_title")
    plt.show()


def test_plot_multiple_ended_with_shots(sample_game):
    game, events = sample_game

    possessions = extract_possessions_ended_with_shots(game, events)

    assert len(possessions) == 10

    fig = plot_multiple_possessions(possessions.values(), main_title=f"{game_summary(game)}\nPossessions ended with Shots")
    fig.savefig('data/test/visualization_ended_with_shots.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_plot_multiple_goals(sample_game):
    game, events = sample_game

    possessions = extract_possessions_ended_with_goals(game, events)

    assert len(possessions) == 3

    fig = plot_multiple_possessions(possessions.values(), main_title=f"{game_summary(game)}\nPossessions with Goals")
    fig.savefig('data/test/visualization_goals.png', dpi=300, bbox_inches='tight')
    plt.show()
