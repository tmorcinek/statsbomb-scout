import warnings

import socceraction.spadl as spadl

from src.analysis.actions_possessions import possessions_from_actions
from src.ml.preprocessing.possessions_extraction import extract_actions_from_events

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_loader import load_statsbomb_socceraction_data, load_socceraction_data


def test_first_match_possessions():
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))

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
