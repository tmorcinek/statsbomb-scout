import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import socceraction.spadl.config as spadl_config
from mplsoccer.soccer.pitch import VerticalPitch
from pandas import DataFrame

from src.analysis.game_utils import get_action_outcomes

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def plot_possession_actions(possession_actions: pd.DataFrame, title: Optional[str] = None, ax=None) -> Optional[plt.Figure]:
    pitch = VerticalPitch(pitch_type='custom', pitch_length=spadl_config.field_length, pitch_width=spadl_config.field_width,
                          pitch_color='white', line_color='black')
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 8))
    else:
        fig = None

    possession_team_id = possession_actions.iloc[0]['possession_team_id']

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

        # print(f"idx: {idx}, Action: {action['type_name']}, Start: ({start_x}, {start_y}), End: ({end_x}, {end_y})")
        # Determine color: black for first action, blue for possessing team, red for opponent
        if idx == 0:
            color = 'black'
        elif action['team_id'] == possession_team_id:
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

    ax.set_title(title or _default_title(possession_actions), fontsize=12, fontweight='normal', pad=20)

    return fig if fig is not None else ax.figure


def _default_title(possession_actions: DataFrame) -> str:
    first_action = possession_actions.iloc[0]
    last_action = possession_actions.iloc[-1]
    possession_length = len(possession_actions)
    period_id = first_action['period_id']

    period_offsets = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    offset = period_offsets.get(period_id, 0) * 60
    time_start_seconds = int(first_action['time_seconds']) + offset
    time_end_seconds = int(last_action['time_seconds']) + offset
    formatted = f"{time_start_seconds // 60:02d}:{time_start_seconds % 60:02d}"
    end_formatted = f"{time_end_seconds // 60:02d}:{time_end_seconds % 60:02d}"

    period_names = {
        1: "1st Half",
        2: "2nd Half",
        3: "Extra Time 1st Half",
        4: "Extra Time 2nd Half",
        5: "Penalty Shootout"
    }
    period_name = period_names.get(period_id, "Unknown")
    outcome = get_action_outcomes(possession_actions)

    return f"PID #{first_action['possession']} ({possession_length}↔), {first_action['possession_team_name']}, {period_name}, {formatted} - {end_formatted}\n{outcome}"


def _title_with_value(possession_actions: DataFrame, predicted_value: float, attention_weights: np.ndarray = None) -> str:
    title = f"{_default_title(possession_actions)}\nValue: {predicted_value:.3f}"

    if attention_weights is not None:
        weights_str = ', '.join([f"{w:.3f}" for w in attention_weights])
        title += f"\nAttention: [{weights_str}]"

    return title


def plot_multiple_possessions(possession_actions_list: list[pd.DataFrame], titles: Optional[list[str]] = None, main_title: Optional[str] = None) -> plt.Figure:
    num_possessions = len(possession_actions_list)

    if num_possessions == 0:
        raise ValueError("possession_actions_list cannot be empty")

    if titles is not None and len(titles) != num_possessions:
        raise ValueError(f"titles length ({len(titles)}) must match possession_actions_list length ({num_possessions})")

    cols = math.ceil(math.sqrt(num_possessions)) + 1
    rows = math.ceil(num_possessions / cols)

    figsize = (6 * cols, 6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_possessions == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for plot_idx, possession_actions in enumerate(possession_actions_list):
        pitch = VerticalPitch(pitch_type='custom', pitch_length=spadl_config.field_length, pitch_width=spadl_config.field_width, pitch_color='white',
                              line_color='black')
        pitch.draw(ax=axes[plot_idx])

        plot_possession_actions(possession_actions, title=titles[plot_idx] if titles else None, ax=axes[plot_idx])

    for idx in range(num_possessions, rows * cols):
        axes[idx].axis('off')

    if main_title is not None:
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.99)

    plt.tight_layout()
    return fig
