import pandas as pd


def game_summary(game: pd.Series) -> str:
    home_team = game['home_team_name']
    away_team = game['away_team_name']
    home_score = game['home_score']
    away_score = game['away_score']
    formatted_date = game['game_date'].strftime('%d.%m.%Y')

    return f"{home_team} {home_score} : {away_score} {away_team} ({game['competition_stage']}) [{formatted_date}]"


def get_action_outcomes(possession_actions: pd.DataFrame) -> str:
    has_goal = False
    has_shot = False
    has_foul = False

    for _, action in possession_actions.iterrows():
        action_type = action['type_id']

        if action_type in [11, 12, 13]:
            if action.get('result_id') == 1:
                has_goal = True
            else:
                has_shot = True
        elif action_type == 8:
            has_foul = True

    if has_goal:
        return "[Goal]"
    elif has_shot:
        return "[Shot]"
    elif has_foul:
        return "[Foul]"
    else:
        return "[Pass]"
