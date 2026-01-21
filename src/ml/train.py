import json
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import config


class ModelTrainer:
    """Handles model training, evaluation, and saving."""

    VALUE_OUTPUT = 'value'
    ATTENTION_OUTPUT = 'attention_weights'

    def __init__(self, model: keras.Model, model_dir: Path = Path(f"models/{config.MODEL_TYPE}/")):
        self.model = model
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        self._is_multi_output = isinstance(model.output, dict)

    def _prepare_labels(self, y: np.ndarray, X_shape: int) -> Union[Dict, np.ndarray]:
        if not self._is_multi_output:
            return y
        return {
            self.VALUE_OUTPUT: y,
            self.ATTENTION_OUTPUT: np.zeros((len(y), X_shape))
        }

    def _get_metric_key(self, history_dict: Dict, primary: str, fallback: str) -> str:
        return next((k for k in [primary, fallback] if k in history_dict), fallback)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 32, epochs: int = 50):

        # Prepare labels for multi-output models
        y_train = self._prepare_labels(y_train, X_train.shape[1])
        y_val = self._prepare_labels(y_val, X_val.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "best_model.keras"),
                monitor='val_loss',
                save_best_only=True
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def _extract_value_predictions(self, predictions):
        return predictions[self.VALUE_OUTPUT].flatten() if isinstance(predictions, dict) else predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        y_test_prepared = self._prepare_labels(y_test, X_test.shape[1])
        results = self.model.evaluate(X_test, y_test_prepared, verbose=0)

        metrics = {
            'test_loss': results[0] if isinstance(results, list) else results,
            'test_mae': results[1] if isinstance(results, list) and len(results) > 1 else results[0]
        }

        y_pred = self._extract_predictions(self.model.predict(X_test))
        metrics['test_rmse'] = np.sqrt(np.mean((y_test - y_pred) ** 2))

        if verbose:
            print(f"\nTest Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        return metrics

    def plot_training_history(self, filename: str = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        history_dict = self.history.history

        loss_key = self._get_metric_key(history_dict, 'value_loss', 'loss')
        val_loss_key = self._get_metric_key(history_dict, 'val_value_loss', 'val_loss')
        mae_key = self._get_metric_key(history_dict, 'value_mae', 'mae')
        val_mae_key = self._get_metric_key(history_dict, 'val_value_mae', 'val_mae')

        ax1.plot(history_dict[loss_key], label='Train Loss')
        ax1.plot(history_dict[val_loss_key], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history_dict[mae_key], label='Train MAE')
        ax2.plot(history_dict[val_mae_key], label='Val MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if filename:
            plt.savefig(self.model_dir / filename)
        plt.show()

    def save_training_metrics(self, metrics: Dict, filename: str = "metrics.json"):
        save_path = self.model_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")
