"""
Model Training Script - Train default model configurations

Usage:
    python train_models.py

This script trains multiple model configurations with predefined hyperparameters
and saves each model in a separate directory with descriptive names.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

import config
from src.data.data_loader import load_statsbomb_socceraction_data
from src.data.data_splitter import split_matches
from src.ml.models.model_factory import create_model
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.train import ModelTrainer
from src.ml.xthreat import get_default_xt_model


class ModelConfig:
    """Configuration for a single model training run."""

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
            self.output_dir = f"models/{model_type}_{name}_{timestamp}"
        else:
            self.output_dir = output_dir

    def __repr__(self):
        return (f"ModelConfig(name='{self.name}', type='{self.model_type}', "
                f"params={self.model_params}, output='{self.output_dir}')")


class ModelTrainingPipeline:
    """Pipeline for training multiple model configurations."""

    def __init__(
        self,
        competition_id: int = 55,
        season_id: int = 282,
        data_dir: str = "data/statsbomb/data"
    ):
        self.competition_id = competition_id
        self.season_id = season_id
        self.data_dir = data_dir

        # Data placeholders
        self.data = None
        self.train_matches = None
        self.val_matches = None
        self.test_matches = None

        self.X_train = None
        self.y_train = None
        self.p_train = None
        self.m_train = None

        self.X_val = None
        self.y_val = None
        self.p_val = None
        self.m_val = None

        self.X_test = None
        self.y_test = None
        self.p_test = None
        self.m_test = None

        self.preprocessor = None

    def load_and_split_data(self):
        """Load data and split into train/val/test sets."""
        print(f"\n{'='*80}")
        print(f"Loading data: Competition {self.competition_id}, Season {self.season_id}")
        print(f"{'='*80}")

        self.data = load_statsbomb_socceraction_data(
            self.data_dir,
            self.competition_id,
            self.season_id
        )

        self.train_matches, self.val_matches, self.test_matches = split_matches(self.data)

        print(f"✓ Train matches: {len(self.train_matches)}")
        print(f"✓ Val matches: {len(self.val_matches)}")
        print(f"✓ Test matches: {len(self.test_matches)}")

    def preprocess_data(
        self,
        sequence_length: Optional[int] = None,
        minimum_sequence_length: Optional[int] = None
    ):
        """
        Preprocess data into sequences.

        This method is called ONCE and caches the processed data.
        All subsequent model training runs will reuse this cached data.
        """
        # Check if data is already preprocessed
        if self.X_train is not None:
            print(f"\n{'='*80}")
            print("Data already preprocessed - reusing cached data")
            print(f"{'='*80}")
            print(f"✓ Train set: {self.X_train.shape[0]} sequences, shape {self.X_train.shape}")
            print(f"✓ Val set: {self.X_val.shape[0]} sequences, shape {self.X_val.shape}")
            print(f"✓ Test set: {self.X_test.shape[0]} sequences, shape {self.X_test.shape}")
            return

        print(f"\n{'='*80}")
        print("Preprocessing data (this will be done only once)")
        print(f"{'='*80}")

        seq_len = sequence_length or config.SEQUENCE_LENGTH
        min_seq_len = minimum_sequence_length or config.MINIMUM_SEQUENCE_LENGTH

        # Create preprocessor instance ONCE
        self.preprocessor = SequencePreprocessor(
            sequence_length=seq_len,
            minimum_sequence_length=min_seq_len,
            xt_model=get_default_xt_model()
        )

        print(f"Processing training set...")
        self.X_train, self.y_train, self.p_train, self.m_train = \
            self.preprocessor.process_matches(self.train_matches)

        print(f"Processing validation set...")
        self.X_val, self.y_val, self.p_val, self.m_val = \
            self.preprocessor.process_matches(self.val_matches)

        print(f"Processing test set...")
        self.X_test, self.y_test, self.p_test, self.m_test = \
            self.preprocessor.process_matches(self.test_matches)

        print(f"\n✓ Train set: {self.X_train.shape[0]} sequences, shape {self.X_train.shape}")
        print(f"✓ Val set: {self.X_val.shape[0]} sequences, shape {self.X_val.shape}")
        print(f"✓ Test set: {self.X_test.shape[0]} sequences, shape {self.X_test.shape}")
        print(f"✓ Data cached in memory - will be reused for all models")

    def train_model(self, model_config: ModelConfig) -> Dict:
        """Train a single model with given configuration."""
        # Validate that data is preprocessed
        if self.X_train is None:
            raise RuntimeError(
                "Data not preprocessed. Call preprocess_data() before training models."
            )

        print(f"\n{'='*80}")
        print(f"Training Model: {model_config.name}")
        print(f"Type: {model_config.model_type}")
        print(f"Output: {model_config.output_dir}")
        print(f"{'='*80}")

        # Create output directory
        Path(model_config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = Path(model_config.output_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'name': model_config.name,
                'model_type': model_config.model_type,
                'model_params': model_config.model_params,
                'training_params': model_config.training_params,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"✓ Saved configuration to {config_path}")

        # Build model
        print(f"\nBuilding {model_config.model_type} model...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        model = create_model(
            model_config.model_type,
            input_shape=input_shape,
            **model_config.model_params
        )

        print(f"✓ Model built with input shape {input_shape}")

        # Train model
        print(f"\nTraining model...")
        trainer = ModelTrainer(model, model_config.output_dir)

        batch_size = model_config.training_params.get('batch_size', config.BATCH_SIZE)
        epochs = model_config.training_params.get('epochs', config.EPOCHS)

        trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            batch_size=batch_size,
            epochs=epochs
        )

        # Evaluate model
        print(f"\nEvaluating model...")
        val_metrics = trainer.evaluate(self.X_val, self.y_val)
        test_metrics = trainer.evaluate(self.X_test, self.y_test)

        print(f"\n{'='*80}")
        print(f"Results for {model_config.name}:")
        print(f"{'='*80}")
        print(f"Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"\nTest Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}")

        # Save metrics
        all_metrics = {
            'validation': val_metrics,
            'test': test_metrics,
            'config': {
                'name': model_config.name,
                'model_type': model_config.model_type,
                'model_params': model_config.model_params,
                'training_params': model_config.training_params
            }
        }

        trainer.save_training_metrics(all_metrics, filename="metrics.json")
        trainer.plot_training_history(filename="training_history.png")

        print(f"\n✓ Model saved to {model_config.output_dir}")
        print(f"✓ Best model: {model_config.output_dir}/best_model.keras")

        return all_metrics

    def train_multiple_models(self, model_configs: List[ModelConfig]) -> pd.DataFrame:
        """Train multiple models and return comparison DataFrame."""
        results = []

        print(f"\n{'='*80}")
        print(f"Training {len(model_configs)} models")
        print(f"{'='*80}")

        for i, model_config in enumerate(model_configs, 1):
            print(f"\n\n{'#'*80}")
            print(f"# Model {i}/{len(model_configs)}: {model_config.name}")
            print(f"{'#'*80}")

            try:
                metrics = self.train_model(model_config)

                results.append({
                    'name': model_config.name,
                    'model_type': model_config.model_type,
                    'output_dir': model_config.output_dir,
                    'val_loss': metrics['validation']['loss'],
                    'val_mae': metrics['validation']['mae'],
                    'val_rmse': metrics['validation']['rmse'],
                    'test_loss': metrics['test']['loss'],
                    'test_mae': metrics['test']['mae'],
                    'test_rmse': metrics['test']['rmse'],
                    'status': 'success'
                })

            except Exception as e:
                print(f"\n❌ Error training {model_config.name}: {e}")
                results.append({
                    'name': model_config.name,
                    'model_type': model_config.model_type,
                    'output_dir': model_config.output_dir,
                    'val_loss': np.nan,
                    'val_mae': np.nan,
                    'val_rmse': np.nan,
                    'test_loss': np.nan,
                    'test_mae': np.nan,
                    'test_rmse': np.nan,
                    'status': f'failed: {str(e)}'
                })

        results_df = pd.DataFrame(results)

        # Save comparison
        comparison_file = "models/model_comparison.csv"
        results_df.to_csv(comparison_file, index=False)
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {comparison_file}")
        print(f"\n{results_df.to_string()}")

        return results_df


def create_default_configs() -> List[ModelConfig]:
    """Create default model configurations for comparison."""
    configs = [
        # LSTM models
        ModelConfig(
            name="lstm_baseline",
            model_type="lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="lstm_large",
            model_type="lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3},
            training_params={'batch_size': 32, 'epochs': 50}
        ),

        # Attention LSTM models
        ModelConfig(
            name="attention_lstm_baseline",
            model_type="attention_lstm",
            model_params={'lstm_units': 64, 'dropout': 0.2},
            training_params={'batch_size': 32, 'epochs': 50}
        ),
        ModelConfig(
            name="attention_lstm_large",
            model_type="attention_lstm",
            model_params={'lstm_units': 128, 'dropout': 0.3},
            training_params={'batch_size': 32, 'epochs': 50}
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
            training_params={'batch_size': 32, 'epochs': 50}
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
            training_params={'batch_size': 32, 'epochs': 50}
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
            training_params={'batch_size': 32, 'epochs': 50}
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
            training_params={'batch_size': 32, 'epochs': 50}
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
            training_params={'batch_size': 32, 'epochs': 50}
        ),
    ]

    return configs


def main():
    """Train default model configurations."""
    # Initialize pipeline
    pipeline = ModelTrainingPipeline(
        competition_id=55,
        season_id=282
    )

    # Load and preprocess data ONCE - will be cached and reused for all models
    print("\n" + "="*80)
    print("STEP 1: Loading and preprocessing data (done once)")
    print("="*80)
    pipeline.load_and_split_data()
    pipeline.preprocess_data()

    print("\n" + "="*80)
    print("STEP 2: Preparing model configurations")
    print("="*80)

    # Use default configurations
    print("\nUsing default model configurations")
    model_configs = create_default_configs()

    # Train models (data is already cached in memory)
    print("\n" + "="*80)
    print("STEP 3: Training models (using cached data)")
    print("="*80)
    results_df = pipeline.train_multiple_models(model_configs)

    print(f"\n{'='*80}")
    print("All models trained successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
