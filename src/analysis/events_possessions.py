import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')

from src.data.data_loader import load_statsbomb_socceraction_data, _filter_relevant_events

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def plot_possession_distribution(possessions_df: pd.DataFrame, title: str = "Rozkład długości possession") -> None:
    """Create a histogram and KDE plot for possession length distribution"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram with KDE
    sns.histplot(data=possessions_df, x='length', kde=True, bins=50, ax=axes[0], color='skyblue')
    axes[0].set_xlabel('Długość possession (liczba zdarzeń)')
    axes[0].set_ylabel('Częstość')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(alpha=0.3)

    # Box plot
    sns.boxplot(data=possessions_df, x='length', ax=axes[1], color='lightcoral')
    axes[1].set_xlabel('Długość possession (liczba zdarzeń)')
    axes[1].set_title(f'{title} - Box plot')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_possession_by_team(possessions_df: pd.DataFrame) -> None:
    """Create a plot showing possession length distribution by team"""
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(data=possessions_df, x='possession_team', y='length', hue='possession_team', ax=ax, palette='Set2', legend=False)
    ax.set_xlabel('Drużyna')
    ax.set_ylabel('Długość possession (liczba zdarzeń)')
    ax.set_title('Rozkład długości possession według drużyny')
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
    return {
        'mean_length': possessions_df['length'].mean(),
        'median_length': possessions_df['length'].median(),
        'std_length': possessions_df['length'].std(),
        'min_length': possessions_df['length'].min(),
        'max_length': possessions_df['length'].max(),
        'total_possessions': len(possessions_df)
    }


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
    all_possessions = analyze_all_matches()
    all_possessions.to_csv('data/processed/events_possessions.csv', index=False)
    # all_possessions = pd.read_csv('data/processed/events_possessions.csv')

    for key, value in calculate_possession_statistics(all_possessions).items():
        print(f"  {key}: {value:.2f}")

    fig1 = plot_possession_distribution(all_possessions)
    fig1.savefig('data/plot/possession_distribution.png', dpi=300, bbox_inches='tight')

    fig2 = plot_possession_by_team(all_possessions)
    fig2.savefig('data/plot/possession_by_team.png', dpi=300, bbox_inches='tight')

    plt.show()
