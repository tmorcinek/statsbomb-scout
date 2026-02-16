import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

from analyse_models import ModelConfig, LoadMatchStep, PreprocessMatchStep, MatchData
from src.ml.models.model_factory import load_model
from src.ml.preprocessing.sequence_preprocessor import PreprocessingMode
from src.analysis.prediction_utils import normalize_predictions
from src.training.base import PipelineStep, Pipeline, BranchingPipeline


class AnalyzePlayersStep(PipelineStep):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    def process(self, data: MatchData) -> Dict:
        print(f"\n{'=' * 80}")
        print(f"Analyzing players with model: {self.model_config.name}")
        print(f"  Type: {self.model_config.model_type}")
        print(f"  Directory: {self.model_config.model_dir}")
        print(f"{'=' * 80}\n")

        print(f"Loading model from {self.model_config.model_dir}...")
        model = load_model(self.model_config.model_type, self.model_config.model_dir)

        print(f"Making predictions on {data.X.shape[0]} sequences...")
        predictions = model.predict(data.X, batch_size=32, verbose=0)
        predicted_values, attention_weights = normalize_predictions(predictions)

        print(f"Predictions complete: {len(predicted_values)} sequences")

        print(f"Calculating player contributions...")
        player_stats = self._calculate_player_contributions(data.p, predicted_values, attention_weights)

        output_dir = Path(self.model_config.model_dir)
        csv_path = output_dir / f"player_contributions_match_{data.match_id}.csv"

        print(f"Saving player contributions to {csv_path}...")
        player_stats_df = pd.DataFrame(player_stats).sort_values('total_value', ascending=False)
        player_stats_df.to_csv(csv_path, index=False)
        print(f"CSV saved: {len(player_stats_df)} players")

        return {
            'model_name': self.model_config.name,
            'model_type': self.model_config.model_type,
            'model_dir': self.model_config.model_dir,
            'match_id': data.match_id,
            'num_sequences': len(predicted_values),
            'num_players': len(player_stats_df),
            'csv_path': str(csv_path),
            'status': 'success'
        }

    def _calculate_player_contributions(self, possessions: np.ndarray, predicted_values: np.ndarray,
                                       attention_weights: Optional[np.ndarray]) -> List[Dict]:
        player_contributions = defaultdict(lambda: {
            'total_value': 0.0,
            'num_sequences': 0,
            'avg_value': 0.0,
            'max_value': 0.0,
            'sequences': []
        })

        for seq_idx, (possession_df, pred_value) in enumerate(zip(possessions, predicted_values)):
            own_team_actions = possession_df[possession_df['opposite_action'] == 0]

            if len(own_team_actions) == 0:
                continue

            for action_idx, action in own_team_actions.iterrows():
                player_id = action['player_id']
                player_name = action['player_name']
                team_name = action['team_name']

                if pd.isna(player_id):
                    continue

                weight = 1.0 / len(own_team_actions)

                if attention_weights is not None and seq_idx < len(attention_weights):
                    action_position = possession_df.index.get_loc(action_idx)
                    if action_position < len(attention_weights[seq_idx]):
                        weight = attention_weights[seq_idx][action_position]

                contribution = pred_value * weight

                player_contributions[player_id]['player_id'] = int(player_id)
                player_contributions[player_id]['player_name'] = player_name
                player_contributions[player_id]['team_name'] = team_name
                player_contributions[player_id]['total_value'] += contribution
                player_contributions[player_id]['num_sequences'] += 1
                player_contributions[player_id]['max_value'] = max(
                    player_contributions[player_id]['max_value'],
                    contribution
                )
                player_contributions[player_id]['sequences'].append({
                    'seq_idx': seq_idx,
                    'value': float(pred_value),
                    'contribution': float(contribution),
                    'weight': float(weight)
                })

        results = []
        for player_id, stats in player_contributions.items():
            stats['avg_value'] = stats['total_value'] / stats['num_sequences'] if stats['num_sequences'] > 0 else 0.0
            del stats['sequences']
            results.append(stats)

        return results


class SavePlayerAnalysisResultsStep(PipelineStep):

    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = output_file

    def process(self, results: Dict) -> Dict:
        print(f"\nSaving player analysis results to {self.output_file}...")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved successfully")
        return results


def create_player_analysis_pipeline(
        match_id: int,
        model_configs: List[ModelConfig],
        data_dir: str = "data/statsbomb/data",
        competition_id: int = 55,
        season_id: int = 282,
        output_file: Optional[str] = None
) -> Pipeline:

    if output_file is None:
        output_file = f"analysis_results/player_analysis_match_{match_id}.json"

    model_branches = {
        cfg.name: AnalyzePlayersStep(model_config=cfg)
        for cfg in model_configs
    }

    pipeline = Pipeline(name=f"PlayerAnalysisPipeline_Match{match_id}")
    pipeline.add_step(LoadMatchStep(
        match_id=match_id,
        data_dir=data_dir,
        competition_id=competition_id,
        season_id=season_id
    ))
    pipeline.add_step(PreprocessMatchStep(mode=PreprocessingMode.PLAYER_EVALUATION))
    pipeline.add_step(BranchingPipeline(name="PlayerAnalysisBranches", branches=model_branches))
    pipeline.add_step(SavePlayerAnalysisResultsStep(output_file))

    return pipeline


if __name__ == "__main__":
    models = [
        ModelConfig(model_type='attention_lstm', model_dir='generated/attention_lstm_attention_lstm_large_20260214_170158'),
        ModelConfig(model_type='transformer', model_dir='generated/transformer_transformer_baseline_20260214_170158'),
    ]

    pipeline = create_player_analysis_pipeline(
        match_id=3943043,
        model_configs=models
    )

    results = pipeline(None)

    print("\n" + "=" * 80)
    print("PLAYER ANALYSIS COMPLETE")
    print("=" * 80)
    for model_name, result in results.items():
        if isinstance(result, dict) and result.get('status') == 'success':
            print(f"\n{model_name}:")
            print(f"  Players analyzed: {result['num_players']}")
            print(f"  Sequences: {result['num_sequences']}")
            print(f"  CSV: {result['csv_path']}")





