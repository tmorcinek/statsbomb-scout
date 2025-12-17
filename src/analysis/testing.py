import warnings

import socceraction.spadl as spadl

from src.analysis.actions_possessions import enrich_actions_with_event_data, possessions_from_actions

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_loader import load_statsbomb_socceraction_data, load_socceraction_data


def test_first_match_possessions():
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))

    actions = spadl.statsbomb.convert_to_actions(events, game['home_team_id'], xy_fidelity_version=2)
    actions = enrich_actions_with_event_data(actions, events)

    possessions = possessions_from_actions(actions)

    print(f"Event columns:\n{len(possessions)}")
    print(f"Events:\n{possessions}")


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
