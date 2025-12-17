import warnings

import matplotlib.pyplot as plt
import pandas as pd
import socceraction.spadl as spadl

from src.analysis.events_possessions import calculate_possession_statistics, calculate_possession_percentiles, plot_possession_distribution, \
    plot_possession_by_team, plot_possession_percentiles

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_loader import load_statsbomb_socceraction_data

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def possessions_from_actions(actions_df: pd.DataFrame) -> pd.DataFrame:
    return actions_df.groupby('possession').agg(
        game_id=('game_id', 'first'),
        possession_team=('team_name', 'first'),
        length=('possession', 'size'),
        start_x=('start_x', 'first'),
        start_y=('start_y', 'first'),
        end_x=('end_x', 'last'),
        end_y=('end_y', 'last'),
        period_id=('period_id', 'first'),
        start_time_seconds=('time_seconds', 'first'),
        end_time_seconds=('time_seconds', 'last')
    ).reset_index()


def extract_season_possessions() -> pd.DataFrame:
    def process_match(game_events):
        game, events = game_events
        actions = spadl.statsbomb.convert_to_actions(events, game['home_team_id'], xy_fidelity_version=2)
        actions = enrich_actions_with_event_data(actions, events)
        return possessions_from_actions(actions)

    return pd.concat(
        map(process_match, load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)),
        ignore_index=True
    )


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


if __name__ == '__main__':
    # possessions = extract_season_possessions()
    # possessions.to_csv('data/processed/actions_possessions.csv', index=False)
    possessions = pd.read_csv('data/processed/actions_possessions.csv')

    stats = calculate_possession_statistics(possessions)
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Percentyle possession (minimalną długość) ---")
    percentiles = calculate_possession_percentiles(possessions)
    for length, pct in percentiles.items():
        print(f"  Co najmniej {length} zdarzeń: {pct:.1f}%")

    fig1 = plot_possession_distribution(possessions)
    fig1.savefig('data/plot/action_possession_distribution.png', dpi=300, bbox_inches='tight')

    fig2 = plot_possession_by_team(possessions)
    fig2.savefig('data/plot/action_possession_by_team.png', dpi=300, bbox_inches='tight')

    fig3 = plot_possession_percentiles(possessions)
    fig3.savefig('data/plot/action_possession_percentiles.png', dpi=300, bbox_inches='tight')

    plt.show()
