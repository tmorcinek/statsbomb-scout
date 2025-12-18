from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import socceraction.spadl as spadl
from mplsoccer import Pitch

from src.analysis.actions_possessions import enrich_actions_with_event_data
from src.data.data_loader import load_statsbomb_socceraction_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def plot_possession_actions(possession_actions: pd.DataFrame, figsize: tuple = (10, 8)) -> Optional[plt.Figure]:
    # Create pitch
    pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68, pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # Get team and color information
    team_name = possession_actions.iloc[0]['team_name']
    colors = possession_actions['team_name'].map(lambda x: 'red' if x == team_name else 'blue')

    # Plot passes/actions as arrows
    for idx, (i, action) in enumerate(possession_actions.iterrows()):
        start_x = action['start_x']
        start_y = action['start_y']
        end_x = action['end_x']
        end_y = action['end_y']
        color = colors.iloc[idx]

        # Draw arrow for each action
        pitch.arrows(
            start_x, start_y, end_x, end_y,
            ax=ax, width=2, color=color, alpha=0.6, label=action['type_name'] if idx == 0 else ""
        )

        # Add circle at start position
        if idx == 0:
            # First action - larger with black edge
            ax.plot(start_x, start_y, 'o', color=color, markersize=10, alpha=0.8, markeredgecolor='black', markeredgewidth=2)
        elif idx == len(possession_actions) - 1:
            # Last action - larger with black edge
            ax.plot(start_x, start_y, 'o', color=color, markersize=10, alpha=0.8, markeredgecolor='black', markeredgewidth=2)
        else:
            # Other actions - regular size
            ax.plot(start_x, start_y, 'o', color=color, markersize=6, alpha=0.8)

    # Add title with possession information
    possession_length = len(possession_actions)
    first_action = possession_actions.iloc[0]
    last_action = possession_actions.iloc[-1]
    period_id = first_action['period_id']
    time_start_seconds = int(first_action['time_seconds'])
    time_end_seconds = int(last_action['time_seconds'])
    time_start_formatted = f"{time_start_seconds // 60:02d}:{time_start_seconds % 60:02d}"
    time_end_formatted = f"{time_end_seconds // 60:02d}:{time_end_seconds % 60:02d}"

    ax.set_title(
        f"Possession #{first_action['possession']} - Team: {team_name} - Actions: {possession_length} - Half: {period_id} - Time: {time_start_formatted} - {time_end_formatted}",
        fontsize=14, fontweight='bold', pad=20
    )

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    return fig


def normalize_pitch(actions: pd.DataFrame) -> pd.DataFrame:
    pitch_length = 105
    pitch_width = 68

    # Rotate start position
    actions['start_x'] = pitch_length - actions['start_x']
    actions['start_y'] = pitch_width - actions['start_y']

    # Rotate end position
    actions['end_x'] = pitch_length - actions['end_x']
    actions['end_y'] = pitch_width - actions['end_y']

    return actions

if __name__ == '__main__':
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))
    print(f"Game info: {game['home_team_id']} vs {game['away_team_id']}\n Game: {game}")
    actions = spadl.statsbomb.convert_to_actions(events, None, xy_fidelity_version=2)
    actions = enrich_actions_with_event_data(actions, events)
    actions = normalize_pitch(actions)

    # Example: Plot a possession
    possession_id = 8
    possession_actions = actions[actions['possession'] == possession_id]
    print(f"Possession actions:\n{possession_actions}")

    events = events.drop(
        columns=['play_pattern_id', 'play_pattern_name',
                 # 'extra',
                 'minute', 'second',
                 'player_id',
                 # 'related_events',
                 'player_name', 'position_id',
                 'visible_area_360',
                 'freeze_frame_360',
                 ],
        errors='ignore'
    )
    possession_events = events[events['event_id'] == '04b74e76-6803-489d-9612-ce3b580fbcff']
    print(f"Possession events:\n{possession_events}")

    fig = plot_possession_actions(possession_actions)
    if fig:
        plt.show()
