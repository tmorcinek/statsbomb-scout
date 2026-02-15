import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.analysis.prediction_utils import normalize_predictions, create_top_sequences, visualize_top_sequences
from src.data.data_loader import load_socceraction_match
from src.ml.models.model_factory import load_model
from src.ml.preprocessing.sequence_preprocessor import SequencePreprocessor, PreprocessingMode
from src.ml.preprocessing.xthreat import get_default_xt_model
from src.training.base import PipelineStep, Pipeline, BranchingPipeline


class ModelConfig:

    def __init__(self, model_type: str, model_dir: str):
        self.model_type = model_type
        self.model_dir = model_dir
        self.name = Path(model_dir).name


class MatchData:

    def __init__(self, match_id: int, X: np.ndarray, y: np.ndarray, p: np.ndarray):
        self.match_id = match_id
        self.X = X
        self.y = y
        self.p = p


class LoadMatchStep(PipelineStep):

    def __init__(self, match_id: int, data_dir: str = "data/statsbomb/data",
                 competition_id: int = 55, season_id: int = 282):
        super().__init__()
        self.match_id = match_id
        self.data_dir = data_dir
        self.competition_id = competition_id
        self.season_id = season_id

    def process(self, data: Any = None) -> Tuple[int, pd.Series, pd.DataFrame]:
        print(f"Loading match {self.match_id} from competition {self.competition_id}, season {self.season_id}...")
        match, events = load_socceraction_match(
            self.data_dir, self.competition_id, self.season_id, self.match_id
        )
        print(f"Match loaded: {len(events)} events")
        return self.match_id, match, events


class PreprocessMatchStep(PipelineStep):

    def __init__(self, sequence_length: int = config.SEQUENCE_LENGTH,
                 minimum_sequence_length: int = config.MINIMUM_SEQUENCE_LENGTH,
                 mode: PreprocessingMode = PreprocessingMode.TRAINING):
        super().__init__()
        self.sequence_length = sequence_length
        self.minimum_sequence_length = minimum_sequence_length
        self.mode = mode

    def process(self, match_data: Tuple[int, pd.Series, pd.DataFrame]) -> MatchData:
        match_id, match, events = match_data

        print(f"Preprocessing match {match_id}...")
        preprocessor = SequencePreprocessor(
            sequence_length=self.sequence_length,
            minimum_sequence_length=self.minimum_sequence_length,
            xt_model=get_default_xt_model()
        )

        X, y, p = preprocessor.process_match(match_id, match, events, mode=self.mode)

        print(f"Match preprocessed: {X.shape[0]} sequences")
        return MatchData(match_id, X, y, p)


class AnalyzeModelStep(PipelineStep):

    def __init__(self, model_config: ModelConfig, top_n: int = 50, top_k: int = 8, cols: int = 4):
        super().__init__()
        self.model_config = model_config
        self.top_n = top_n
        self.top_k = top_k
        self.cols = cols

    def process(self, data: MatchData) -> Dict:
        print(f"\n{'=' * 80}")
        print(f"Analyzing model: {self.model_config.name}")
        print(f"  Type: {self.model_config.model_type}")
        print(f"  Directory: {self.model_config.model_dir}")
        print(f"{'=' * 80}\n")

        print(f"Loading model from {self.model_config.model_dir}...")
        model = load_model(self.model_config.model_type, self.model_config.model_dir)

        print(f"Making predictions on {data.X.shape[0]} sequences...")
        predictions = model.predict(data.X, batch_size=32, verbose=0)
        predicted_values, attention_weights = normalize_predictions(predictions)

        print(f"Predictions complete:")
        print(f"  Total sequences: {len(predicted_values)}")
        print(f"  Sum of predicted values: {predicted_values.sum():.3f}")
        print(f"  Has attention weights: {attention_weights is not None}")

        output_dir = Path(self.model_config.model_dir)
        csv_path = output_dir / f"top_possessions_match_{data.match_id}.csv"

        print(f"\nSaving top {self.top_n} possessions to {csv_path}...")
        top_possessions_df = create_top_sequences(
            data.p, predicted_values, attention_weights, head=self.top_n
        )
        top_possessions_df.to_csv(csv_path, index=False)
        print(f"CSV saved: {len(top_possessions_df)} possessions")

        image_path = output_dir / f"top_possessions_match_{data.match_id}.png"

        print(f"Visualizing top {self.top_k} possessions and saving to {image_path}...")
        visualize_top_sequences(
            data.p, predicted_values, attention_weights, self.top_k, cols=self.cols
        )
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image saved: {image_path}")

        return {
            'model_name': self.model_config.name,
            'model_type': self.model_config.model_type,
            'model_dir': self.model_config.model_dir,
            'match_id': data.match_id,
            'num_sequences': len(predicted_values),
            'sum_predicted_values': float(predicted_values.sum()),
            'has_attention_weights': attention_weights is not None,
            'top_n': self.top_n,
            'top_k': self.top_k,
            'csv_path': str(csv_path),
            'image_path': str(image_path),
            'status': 'success'
        }


class SaveAnalysisResultsStep(PipelineStep):

    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = output_file

    def process(self, results: Dict) -> Dict:
        print(f"\nSaving analysis results to {self.output_file}...")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved: {len(results)} model(s) analyzed")
        return results


def create_analysis_pipeline(
        match_id: int,
        model_configs: List[ModelConfig],
        top_n: int = 50,
        top_k: int = 8,
        cols: int = 4,
        mode: PreprocessingMode = PreprocessingMode.TRAINING,
        data_dir: str = "data/statsbomb/data",
        competition_id: int = 55,
        season_id: int = 282,
        output_file: Optional[str] = None
) -> Pipeline:
    if output_file is None:
        output_file = f"analysis_results/match_{match_id}_analysis.json"

    model_branches = {
        cfg.name: AnalyzeModelStep(model_config=cfg, top_n=top_n, top_k=top_k, cols=cols)
        for cfg in model_configs
    }

    pipeline = Pipeline(name=f"MatchAnalysisPipeline_Match{match_id}")
    pipeline.add_step(LoadMatchStep(match_id=match_id, data_dir=data_dir, competition_id=competition_id, season_id=season_id))
    pipeline.add_step(PreprocessMatchStep(mode=mode))
    pipeline.add_step(BranchingPipeline(name="ModelAnalysisBranches", branches=model_branches))
    pipeline.add_step(SaveAnalysisResultsStep(output_file))

    return pipeline


if __name__ == "__main__":
    models = [
        # ModelConfig(model_type='attention_lstm', model_dir='generated/attention_lstm_attention_lstm_baseline_20260214_170158'),
        # ModelConfig(model_type='transformer', model_dir='generated/transformer_transformer_small_20260214_170158'),
        ModelConfig(model_type='attention_lstm', model_dir='generated/attention_lstm_attention_lstm_large_20260214_170158'),
        ModelConfig(model_type='transformer', model_dir='generated/transformer_transformer_baseline_20260214_170158'),
    ]

    pipeline = create_analysis_pipeline(
        match_id=3943043,
        model_configs=models,
        top_n=None,
        top_k=20,
        cols=4,
        mode=PreprocessingMode.VALIDATION,
    )

    results = pipeline(None)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    for model_name, result in results.items():
        if isinstance(result, dict) and result.get('status') == 'success':
            print(f"\n{model_name}:")
            print(f"  Sequences analyzed: {result['num_sequences']}")
            print(f"  CSV: {result['csv_path']}")
            print(f"  Image: {result['image_path']}")
