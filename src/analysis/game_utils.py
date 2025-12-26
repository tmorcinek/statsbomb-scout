import pandas as pd


def game_summary(game: pd.Series) -> str:
    home_team = game['home_team_name']
    away_team = game['away_team_name']
    home_score = game['home_score']
    away_score = game['away_score']
    formatted_date = game['game_date'].strftime('%d.%m.%Y')

    return f"{home_team} {home_score} : {away_score} {away_team} ({game['competition_stage']}) [{formatted_date}]"
