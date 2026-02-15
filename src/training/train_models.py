import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from src.data.data_loader import load_statsbomb_socceraction_data
from src.data.data_splitter import split_matches
from src.ml.models.model_factory import create_model
from src.ml.preprocessing.sequence_preprocessor import SequencePreprocessor
from src.ml.train import ModelTrainer
from src.ml.preprocessing.xthreat import get_default_xt_model
from src.training.base import PipelineStep, Pipeline, BranchingPipeline


class ModelConfig:

    def __init__(
            self,
            name: str,
            model_type: str,
            model_params: Dict,
            training_params: Dict,
            output_dir: Optional[str] = None
    ):
        self.name = name
        self.model_type = model_type
        self.model_params = model_params
        self.training_params = training_params

        # Generate output directory name if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"generated/{model_type}_{name}_{timestamp}"
        else:
            self.output_dir = output_dir

    def __repr__(self):
        return (f"ModelConfig(name='{self.name}', type='{self.model_type}', "
                f"params={self.model_params}, output='{self.output_dir}')")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ProcessedData:

    def __init__(self, X_train, y_train, p_train, m_train,
                 X_val, y_val, p_val, m_val,
                 X_test, y_test, p_test, m_test):
        self.X_train = X_train
        self.y_train = y_train
        self.p_train = p_train
        self.m_train = m_train

        self.X_val = X_val
        self.y_val = y_val
        self.p_val = p_val
        self.m_val = m_val

        self.X_test = X_test
        self.y_test = y_test
        self.p_test = p_test
        self.m_test = m_test


# ============================================================================
# PIPELINE STEPS
# ============================================================================

class LoadDataStep(PipelineStep):

    def __init__(self, data_dir: str = "data/statsbomb/data", competition_id: int = 55, season_id: int = 282):
        super().__init__()
        self.data_dir = data_dir
        self.competition_id = competition_id
        self.season_id = season_id

    def process(self, data: Any = None) -> tuple:
        data = load_statsbomb_socceraction_data(self.data_dir, self.competition_id, self.season_id)
        return split_matches(data, random_seed=18)


class PreprocessDataStep(PipelineStep):

    def __init__(self, sequence_length: int = config.SEQUENCE_LENGTH, minimum_sequence_length: int = config.MINIMUM_SEQUENCE_LENGTH):
        super().__init__()
        self.sequence_length = sequence_length
        self.minimum_sequence_length = minimum_sequence_length

    def process(self, matches_data: tuple) -> ProcessedData:
        train_matches, val_matches, test_matches = matches_data

        preprocessor = SequencePreprocessor(
            sequence_length=self.sequence_length,
            minimum_sequence_length=self.minimum_sequence_length,
            xt_model=get_default_xt_model()
        )

        X_train, y_train, p_train, m_train = preprocessor.process_matches(train_matches)
        X_val, y_val, p_val, m_val = preprocessor.process_matches(val_matches)
        X_test, y_test, p_test, m_test = preprocessor.process_matches(test_matches)

        return ProcessedData(
            X_train, y_train, p_train, m_train,
            X_val, y_val, p_val, m_val,
            X_test, y_test, p_test, m_test
        )


class TrainModelStep(PipelineStep):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    def process(self, data: ProcessedData) -> Dict:
        model_config = self.model_config

        Path(model_config.output_dir).mkdir(parents=True, exist_ok=True)

        input_shape = (data.X_train.shape[1], data.X_train.shape[2])
        model = create_model(
            model_config.model_type,
            input_shape=input_shape,
            **model_config.model_params
        )

        trainer = ModelTrainer(model, model_config.output_dir)

        batch_size = model_config.training_params.get('batch_size', config.BATCH_SIZE)
        epochs = model_config.training_params.get('epochs', config.EPOCHS)

        trainer.train(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            batch_size=batch_size,
            epochs=epochs
        )

        val_metrics = trainer.evaluate(data.X_val, data.y_val)
        test_metrics = trainer.evaluate(data.X_test, data.y_test)

        all_metrics = {
            'validation': val_metrics,
            'test': test_metrics,
            'config': {
                'name': model_config.name,
                'model_type': model_config.model_type,
                'model_params': model_config.model_params,
                'training_params': model_config.training_params,
                'timestamp': datetime.now().isoformat()
            }
        }

        trainer.save_training_metrics(all_metrics, filename="metrics.json")
        trainer.plot_training_history(filename="training_history.png")

        return {
            'name': model_config.name,
            'model_type': model_config.model_type,
            'output_dir': model_config.output_dir,
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'test_loss': test_metrics['loss'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'status': 'success'
        }


class SaveResultsStep(PipelineStep):

    def __init__(self, comparison_file: str = "models/model_comparison.csv"):
        super().__init__()
        self.comparison_file = comparison_file

    def process(self, results: Dict) -> Dict:
        results_list = list(results.values())
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(self.comparison_file, index=False)
        return results


def create_default_configs() -> List[ModelConfig]:
    configs = [
        # LSTM models
        ModelConfig(
            name="lstm_small",
            model_type="lstm",
            model_params={'lstm_units': 32, 'dropout': 0.15},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="lstm_baseline",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_large",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3},
            training_params={'batch_size': 32, 'epochs': 100}
        ),

        # Attention LSTM models
        ModelConfig(
            name="attention_lstm_small",
            model_type="attention_lstm",
            model_params={'lstm_units': 32, 'dropout': 0.15},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="attention_lstm_baseline",
            model_type="attention_lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="attention_lstm_large",
            model_type="attention_lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3},
            training_params={'batch_size': 32, 'epochs': 100}
        ),

        # BiGRU models with attention
        ModelConfig(
            name="bigru_baseline",
            model_type="bigru",
            model_params={
                'gru_units': 64,
                'attn_hidden': 32,
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
                'l2_reg': 0.01,
                'learning_rate': 0.0005
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="bigru_medium",
            model_type="bigru",
            model_params={
                'gru_units': 96,
                'attn_hidden': 48,
                'dropout': 0.25,
                'recurrent_dropout': 0.12,
                'l2_reg': 0.012,
                'learning_rate': 0.0004
            },
            training_params={'batch_size': 32, 'epochs': 125}
        ),
        ModelConfig(
            name="bigru_large",
            model_type="bigru",
            model_params={
                'gru_units': 128,
                'attn_hidden': 64,
                'dropout': 0.3,
                'recurrent_dropout': 0.15,
                'l2_reg': 0.015,
                'learning_rate': 0.0003
            },
            training_params={'batch_size': 32, 'epochs': 150}
        ),

        # Transformer models - tanh attention
        ModelConfig(
            name="transformer_baseline",
            model_type="transformer",
            model_params={
                'num_heads': 4,
                'd_model': 128,
                'ff_dim': 512,
                'num_blocks': 2,
                'dropout': 0.1,
                'true_attention': False
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),

        # Transformer models - true attention
        ModelConfig(
            name="transformer_true_attention",
            model_type="transformer",
            model_params={
                'num_heads': 4,
                'd_model': 128,
                'ff_dim': 512,
                'num_blocks': 2,
                'dropout': 0.1,
                'true_attention': True
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),

        # Transformer - small
        ModelConfig(
            name="transformer_small",
            model_type="transformer",
            model_params={
                'num_heads': 2,
                'd_model': 64,
                'ff_dim': 256,
                'num_blocks': 2,
                'dropout': 0.1,
                'true_attention': False
            },
            training_params={'batch_size': 32, 'epochs': 75}
        ),

        # Transformer - large
        ModelConfig(
            name="transformer_large",
            model_type="transformer",
            model_params={
                'num_heads': 8,
                'd_model': 256,
                'ff_dim': 1024,
                'num_blocks': 3,
                'dropout': 0.15,
                'true_attention': True
            },
            training_params={'batch_size': 32, 'epochs': 150}
        ),
    ]

    return configs


def create_lstm_configs() -> List[ModelConfig]:
    configs = [
        ModelConfig(
            name="lstm_small_dense64",
            model_type="lstm",
            model_params={'lstm_units': 32, 'dropout': 0.15, 'dense_units': 64},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="lstm_small_dense128",
            model_type="lstm",
            model_params={'lstm_units': 32, 'dropout': 0.15, 'dense_units': 128},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="lstm_baseline_dense64",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2, 'dense_units': 64},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_baseline_dense128",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2, 'dense_units': 128},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_large_dense64",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'dense_units': 64},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="lstm_large_dense128",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'dense_units': 128},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="lstm_small_no_dense",
            model_type="lstm",
            model_params={'lstm_units': 32, 'dropout': 0.15, 'dense_units': None},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="lstm_baseline_no_dense",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2, 'dense_units': None},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_large_no_dense",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'dense_units': None},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
    ]
    return configs


def create_lstm_second_lstm_configs() -> List[ModelConfig]:
    """Test impact of second LSTM layer with use_second_lstm flag"""
    configs = [
        ModelConfig(
            name="lstm_baseline_single_lstm",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2, 'use_second_lstm': False},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_baseline_double_lstm",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2, 'use_second_lstm': True},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="lstm_large_single_lstm",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'use_second_lstm': False},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="lstm_large_double_lstm",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'use_second_lstm': True},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
    ]
    return configs

def create_transformers_config() -> List[ModelConfig]:
    configs = [
        ModelConfig(
            name="lstm_large_single_lstm",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3, 'use_second_lstm': False},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="transformer_baseline",
            model_type="transformer",
            model_params={
                'num_heads': 4,
                'd_model': 128,
                'ff_dim': 512,
                'num_blocks': 2,
                'dropout': 0.1,
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="transformer_small",
            model_type="transformer",
            model_params={
                'num_heads': 2,
                'd_model': 64,
                'ff_dim': 256,
                'num_blocks': 2,
                'dropout': 0.1,
                'true_attention': False
            },
            training_params={'batch_size': 32, 'epochs': 75}
        ),
    ]
    return configs

def generated_configs() -> List[ModelConfig]:
    configs = [
        # Attention LSTM models
        ModelConfig(
            name="attention_lstm_baseline",
            model_type="attention_lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2},
            training_params={'batch_size': 32, 'epochs': 75}
        ),
        ModelConfig(
            name="attention_lstm_large",
            model_type="attention_lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3},
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        # Transformer models
        ModelConfig(
            name="transformer_small",
            model_type="transformer",
            model_params={
                'num_heads': 2,
                'd_model': 64,
                'ff_dim': 256,
                'num_blocks': 2,
                'dropout': 0.1,
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),
        ModelConfig(
            name="transformer_baseline",
            model_type="transformer",
            model_params={
                'num_heads': 4,
                'd_model': 128,
                'ff_dim': 512,
                'num_blocks': 2,
                'dropout': 0.1,
            },
            training_params={'batch_size': 32, 'epochs': 100}
        ),
    ]
    return configs


def create_comparison_pipeline(
    configs: List[ModelConfig],
    pipeline_name: str = "ComparisonPipeline",
    competition_id: int = 55,
    season_id: int = 282
) -> Pipeline:
    model_branches = {cfg.name: TrainModelStep(model_config=cfg)for cfg in configs}

    pipeline = Pipeline(name=pipeline_name)
    pipeline.add_step(LoadDataStep(competition_id=competition_id, season_id=season_id))
    pipeline.add_step(PreprocessDataStep())
    pipeline.add_step(BranchingPipeline(branches=model_branches))
    pipeline.add_step(SaveResultsStep(f"generated/{pipeline_name}.csv"))

    return pipeline


if __name__ == "__main__":
    lstm_pipeline = create_comparison_pipeline(
        configs=generated_configs(),
        pipeline_name="model_comparison_transformer",
        competition_id=55,
        season_id=282
    )
    lstm_results = lstm_pipeline(None)
