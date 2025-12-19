import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer.soccer.pitch import VerticalPitch

import config
from src.ml.preprocessing.possessions_extraction import extract_possessions_with_shots, extract_possessions
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
        s=40, c=start_colors,
        alpha=0.8, ax=ax
    )

    # Draw first point with black edge on top
    pitch.scatter(
        np.array([start_xs[0]]), np.array([start_ys[0]]),
        s=40, c=[start_colors[0]],
        alpha=0.8, ax=ax, zorder=4
    )

    # Draw star marker at the end of the last action
    pitch.scatter(
        np.array(end_xs), np.array(end_ys),
        s=40, c=['black'], marker='*',
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


def plot_multiple_possessions(possession_actions_list: list[pd.DataFrame], figsize: tuple = None) -> plt.Figure:
    num_possessions = len(possession_actions_list)

    if num_possessions == 0:
        raise ValueError("possession_actions_list cannot be empty")

    cols = math.ceil(math.sqrt(num_possessions)) + 1
    rows = math.ceil(num_possessions / cols)

    if figsize is None:
        figsize = (6 * cols, 6 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_possessions == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for plot_idx, possession_actions in enumerate(possession_actions_list):
        pitch = VerticalPitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, pitch_color='white', line_color='black')
        pitch.draw(ax=axes[plot_idx])

        plot_possession_actions(possession_actions, ax=axes[plot_idx])

    for idx in range(num_possessions, rows * cols):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    game, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))
    home_team_id = game['home_team_id']
    possessions = extract_possessions(game, events)
    # possessions = extract_possessions_with_shots(game, events)
    # possessions = extract_possessions_ended_with_goals(game, events)
    print(f"Total possessions with shots: \n{len(possessions)}")

    possessions_list = [possessions[pid] for pid in sorted(possessions.keys()) if len(possessions[pid]) >= config.SEQUENCE_LENGTH]
    best_possessions = [78, 8, 68, 18, 13]
    possessions_list = [possessions_list[pid] for pid in best_possessions]

    fig = plot_multiple_possessions(possessions_list)
    fig.savefig('data/plot/possessions_best.png', dpi=300, bbox_inches='tight')
    # fig.savefig('data/plot/possessions_67_7_58_15_11.png', dpi=300, bbox_inches='tight')

    plt.show()
