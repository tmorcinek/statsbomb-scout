"""Expected Threat (xT) model utilities for valuing actions."""

import os
import pickle

import pandas as pd
import socceraction.spadl as spadl
import socceraction.xthreat
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.xthreat import load_model, ExpectedThreat


def get_default_xt_model(file_path: str = "models/xt_models/default_xt_model.json") -> ExpectedThreat:
    return load_model(file_path)


def get_xt_model_for_competition(season_id: int, competition_id: int) -> ExpectedThreat:
    SBL = StatsBombLoader(root="data/statsbomb/data", getter="local")
    return get_xt_model("models/xt_models", SBL, season_id, competition_id, l=16, w=12)


def train_xt_model(loader, competition_id, season_id, l=16, w=12):
    """
    Train an Expected Threat (xT) model for a given competition and season.

    Collects games via the provided loader, converts StatsBomb events to SPADL actions,
    normalizes play direction, adds action names, fits an ExpectedThreat model with an
    l-by-w grid, and returns the trained model.

    Args:
        loader: data loader with .games(...) and .events(game_id)
        competition_id (int): StatsBomb competition id
        season_id (int): StatsBomb season id
        l (int): grid cells along the pitch length (default 16)
        w (int): grid cells along the pitch width (default 12)

    Returns:
        ExpectedThreat: trained xT model
    """
    df_games = loader.games(competition_id=competition_id, season_id=season_id)
    dataset = [
        {
            **game,
            'actions': spadl.statsbomb.convert_to_actions(
                events=loader.events(game['game_id']),
                home_team_id=game['home_team_id']
            )
        }
        for game in df_games.to_dict(orient='records')
    ]
    df_actions_ltr = pd.concat([
        spadl.play_left_to_right(game['actions'], game['home_team_id'])
        for game in dataset
    ])
    df_actions_ltr = spadl.add_names(df_actions_ltr)
    xTModel = socceraction.xthreat.ExpectedThreat(l=l, w=w)
    xTModel.fit(df_actions_ltr)
    return xTModel


def get_model_filename(season_id: int, competition_id: int) -> str:
    return f"{season_id}_{competition_id}.pkl"


def save_xt_model(model, filepath, season_id, competition_id):
    """Zapisuje model xT do pliku pickle o nazwie filepath/season_id_competition_id.pkl."""
    os.makedirs(filepath, exist_ok=True)
    model_path = os.path.join(filepath, get_model_filename(season_id, competition_id))
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_xt_model(filepath, season_id, competition_id):
    """Wczytuje model xT z pliku pickle o nazwie filepath/season_id_competition_id.pkl. Zwraca None jeśli nie istnieje."""
    model_path = os.path.join(filepath, get_model_filename(season_id, competition_id))
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_xt_model(filepath, loader, season_id, competition_id, l=16, w=12):
    """Zwraca model xT z pliku jeśli istnieje, w przeciwnym razie trenuje, zapisuje i zwraca."""
    model = load_xt_model(filepath, season_id, competition_id)
    if model is not None:
        return model
    model = train_xt_model(loader, competition_id, season_id, l=l, w=w)
    save_xt_model(model, filepath, season_id, competition_id)
    return model
