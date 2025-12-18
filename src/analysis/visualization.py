from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import socceraction.spadl as spadl
from mplsoccer.soccer.pitch import VerticalPitch

from src.analysis.actions_possessions import enrich_actions_with_event_data
from src.data.data_loader import load_statsbomb_socceraction_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def plot_possession_actions(possession_actions: pd.DataFrame, figsize: tuple = (10, 8)) -> Optional[plt.Figure]:
    # Create pitch
    pitch = VerticalPitch(pitch_type='custom', pitch_length=105, pitch_width=68, pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # Get team and color information
    team_name = possession_actions.iloc[0]['team_name']

    # Separate arrays for scatter plot
    start_xs = []
    start_ys = []
    start_colors = []

    # Arrays for end position marker (star for last action)
    end_xs = []
    end_ys = []

    # Plot passes/actions as arrows
    for idx, (i, action) in enumerate(possession_actions.iterrows()):
        start_x = action['start_x']
        start_y = action['start_y']
        end_x = action['end_x']
        end_y = action['end_y']

        print(f"idx: {idx}, Action: {action['type_name']}, Start: ({start_x}, {start_y}), End: ({end_x}, {end_y})")
        # Determine color: black for first action, blue for possessing team, red for opponent
        if idx == 0:
            color = 'black'
        elif action['team_name'] == team_name:
            color = 'blue'
        else:
            color = 'red'

        # Draw arrow for each action
        pitch.arrows(
            start_x, start_y, end_x, end_y,
            ax=ax, width=2, color=color, alpha=0.6,
            label=action['type_name'] if idx == 0 else ""
        )

        # Collect start positions for scatter plot
        start_xs.append(start_x)
        start_ys.append(start_y)
        start_colors.append(color)

        # Collect end position marker for the last action
        if idx == len(possession_actions) - 1:
            end_xs.append(end_x)
            end_ys.append(end_y)

    print(f" colors for start positions: {start_colors}")
    # Draw all circles using scatter
    pitch.scatter(
        np.array(start_xs), np.array(start_ys),
        s=60, c=start_colors,
        alpha=0.8, ax=ax
    )

    pitch.scatter(
        np.array([start_xs[0]]), np.array([start_ys[0]]),
        s=60, c=[start_colors[0]],
        edgecolors='black', linewidths=2,
        alpha=0.8, ax=ax, zorder=4
    )

    pitch.scatter(
        np.array(end_xs), np.array(end_ys),
        s=np.array(60), c=['black'], marker='*',
        alpha=0.9, ax=ax, zorder=5
    )

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


def normalize_pitch(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    if possession_actions.iloc[0]['team_id'] == home_team_id:
        return actions

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
    home_team_id = game['home_team_id']
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id, xy_fidelity_version=2)
    actions = enrich_actions_with_event_data(actions, events)

    # Example: Plot a possession
    possession_id = 11
    possession_actions = actions[actions['possession'] == possession_id]
    print(f"Possession actions:\n{possession_actions}")

    # possession_actions = possession_actions.tail(10)

    possession_events = normalize_pitch(possession_actions, home_team_id)

    print(f"Possession events:\n{possession_events}")

    fig = plot_possession_actions(possession_actions)
    if fig:
        plt.show()
