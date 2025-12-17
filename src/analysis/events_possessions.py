import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')

from src.data.data_loader import load_statsbomb_socceraction_data, _filter_relevant_events

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def plot_possession_distribution(possessions_df: pd.DataFrame, title: str = "Rozkład długości possession") -> plt.Figure:
    """Create a histogram and KDE plot for possession length distribution"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram with KDE
    sns.histplot(data=possessions_df, x='length', kde=True, bins=200, ax=axes[0], color='skyblue')
    axes[0].set_xlabel('Długość possession (liczba zdarzeń)')
    axes[0].set_ylabel('Częstość')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(alpha=0.9)

    # Box plot
    sns.boxplot(data=possessions_df, x='length', ax=axes[1], color='lightcoral')
    axes[1].set_xlabel('Długość possession (liczba zdarzeń)')
    axes[1].set_title(f'{title} - Box plot')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_possession_by_team(possessions_df: pd.DataFrame) -> plt.Figure:
    """Create a plot showing possession length distribution by team"""
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(data=possessions_df, x='possession_team', y='length', hue='possession_team', ax=ax, palette='Set2', legend=False)
    ax.set_xlabel('Drużyna')
    ax.set_ylabel('Długość possession (liczba zdarzeń)')
    ax.set_title('Rozkład długości possession według drużynie')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig


def analyze_possessions(events_df: pd.DataFrame) -> pd.DataFrame:
    return events_df.groupby('possession').agg(
        game_id=('game_id', 'first'),
        possession_team=('possession_team_name', 'first'),
        length=('possession', 'size'),
        start_position=('location', 'first'),
        end_position=('location', 'last'),
        period_id=('period_id', 'first'),
        start_timestamp=('timestamp', 'first'),
        end_timestamp=('timestamp', 'last')
    ).reset_index()


def calculate_possession_statistics(possessions_df: pd.DataFrame) -> dict:
    mode_result = possessions_df['length'].mode()
    return {
        'mean_length': possessions_df['length'].mean(),
        'median_length': possessions_df['length'].median(),
        'mode_length': mode_result[0] if not mode_result.empty else None,
        'std_length': possessions_df['length'].std(),
        'min_length': possessions_df['length'].min(),
        'max_length': possessions_df['length'].max(),
        'q25': possessions_df['length'].quantile(0.25),
        'q75': possessions_df['length'].quantile(0.75),
        'total_possessions': len(possessions_df)
    }


def calculate_possession_percentiles(possessions_df: pd.DataFrame) -> dict:
    """Calculate what percentage of possessions have >= specific minimum length"""
    percentiles = {}
    max_length = int(possessions_df['length'].max())
    for length in range(2, min(15, max_length + 1)):
        pct = (possessions_df['length'] >= length).sum() / len(possessions_df) * 100
        percentiles[length] = pct
    return percentiles


def plot_possession_percentiles(possessions_df: pd.DataFrame) -> plt.Figure:
    """Create a cumulative distribution plot showing minimum possession lengths"""
    percentiles = calculate_possession_percentiles(possessions_df)
    lengths = list(percentiles.keys())
    percentages = list(percentiles.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lengths, percentages, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(y=95, color='r', linestyle='--', label='95%', linewidth=2)
    ax.axhline(y=90, color='orange', linestyle='--', label='90%', linewidth=1)
    ax.axhline(y=80, color='green', linestyle='--', label='80%', linewidth=1)
    ax.axhline(y=70, color='purple', linestyle='--', label='70%', linewidth=1)
    ax.set_xlabel('Minimalna długość possession (liczba zdarzeń)')
    ax.set_ylabel('Procent wszystkich possessions (%)')
    ax.set_title('Rozkład possessions - jaki procent ma co najmniej X eventów')
    ax.set_xticks(lengths)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim((0, 105))
    plt.tight_layout()
    return fig


def analyze_all_matches() -> pd.DataFrame:
    return pd.concat((
        analyze_possessions(_filter_relevant_events(
            events.assign(type=events["type_name"])
            .sort_values("index", kind="mergesort")
            .pipe(lambda df: df[df['team_id'] == df['possession_team_id']])
        ))
        for _, events in load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
    ), ignore_index=True)


if __name__ == '__main__':
    # all_possessions = analyze_all_matches()
    # all_possessions.to_csv('data/processed/events_possessions.csv', index=False)
    all_possessions = pd.read_csv('data/processed/events_possessions.csv')

    stats = calculate_possession_statistics(all_possessions)
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Percentyle possession (minimalną długość) ---")
    percentiles = calculate_possession_percentiles(all_possessions)
    for length, pct in percentiles.items():
        print(f"  Co najmniej {length} zdarzeń: {pct:.1f}%")

    fig1 = plot_possession_distribution(all_possessions)
    fig1.savefig('data/plot/possession_distribution.png', dpi=300, bbox_inches='tight')

    fig2 = plot_possession_by_team(all_possessions)
    fig2.savefig('data/plot/possession_by_team.png', dpi=300, bbox_inches='tight')

    fig3 = plot_possession_percentiles(all_possessions)
    fig3.savefig('data/plot/possession_percentiles.png', dpi=300, bbox_inches='tight')

    plt.show()
