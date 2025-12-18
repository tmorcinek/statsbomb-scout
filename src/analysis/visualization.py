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

PITCH_LENGTH = 105
PITCH_WIDTH = 68


def plot_possession_actions(possession_actions: pd.DataFrame, ax=None, figsize: tuple = (10, 8)) -> Optional[plt.Figure]:
    # If no axis provided, create a new figure
    if ax is None:
        pitch = VerticalPitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, pitch_color='white', line_color='black')
        fig, ax = pitch.draw(figsize=figsize)
    else:
        pitch = VerticalPitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, pitch_color='white', line_color='black')
        fig = None

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

    # Draw all circles using scatter
    pitch.scatter(
        np.array(start_xs), np.array(start_ys),
        s=60, c=start_colors,
        alpha=0.8, ax=ax
    )

    # Draw first point with black edge on top
    pitch.scatter(
        np.array([start_xs[0]]), np.array([start_ys[0]]),
        s=60, c=[start_colors[0]],
        edgecolors='black', linewidths=2,
        alpha=0.8, ax=ax, zorder=4
    )

    # Draw star marker at the end of the last action
    pitch.scatter(
        np.array(end_xs), np.array(end_ys),
        s=80, c=['black'], marker='*',
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
        f"Possession #{first_action['possession']} - Team: {team_name} - Actions: {possession_length} \n Half: {period_id} - Time: {time_start_formatted} - {time_end_formatted}",
        fontsize=12, fontweight='normal', pad=20
    )

    return fig if fig is not None else ax.figure


def plot_multiple_possessions(actions: pd.DataFrame, possession_ids: list, home_team_id: int, figsize: tuple = (18, 12)) -> plt.Figure:
    if len(possession_ids) > 6:
        print("Warning: Only 6 possessions can be displayed. Showing first 6.")
        possession_ids = possession_ids[:6]

    # Create figure with 2x3 subplots
    pitches = [VerticalPitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                             pitch_color='white', line_color='black') for _ in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten().tolist()

    # Draw each possession
    for plot_idx, possession_id in enumerate(possession_ids):
        possession_actions = actions[actions['possession'] == possession_id]
        possession_actions = normalize_pitch(possession_actions, home_team_id)

        if len(possession_actions) == 0:
            axes[plot_idx].text(0.5, 0.5, f'No actions for possession {possession_id}',
                                ha='center', va='center', transform=axes[plot_idx].transAxes)
            continue

        # Draw pitch
        pitches[plot_idx].draw(ax=axes[plot_idx])

        # Plot possession on this subplot
        plot_possession_actions(possession_actions, ax=axes[plot_idx])  # type: ignore

    # Hide unused subplots
    for idx in range(len(possession_ids), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


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


if __name__ == '__main__':
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))
    home_team_id = game['home_team_id']
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id, xy_fidelity_version=2)
    actions = enrich_actions_with_event_data(actions, events)

    # Example: Plot 6 possessions on one figure
    possession_ids = [7, 8, 9, 10, 11, 12]
    fig = plot_multiple_possessions(actions, possession_ids, home_team_id)
    fig.savefig('data/plot/possessions_7-12_pitch.png', dpi=300, bbox_inches='tight')

    plt.show()
