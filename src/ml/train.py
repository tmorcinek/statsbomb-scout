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

    def __init__(self, model: keras.Model, model_dir: str):
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

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 32, epochs: int = 50):

        # Prepare labels for multi-output models
        y_train = self._prepare_labels(y_train, X_train.shape[1])
        y_val = self._prepare_labels(y_val, X_val.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LEARNING_RATE,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
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

    def _extract_predictions(self, predictions):
        return predictions[self.VALUE_OUTPUT].flatten() if isinstance(predictions, dict) else predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_test_prepared = self._prepare_labels(y_test, X_test.shape[1])
        results = self.model.evaluate(X_test, y_test_prepared, verbose=0)

        if isinstance(results, list):
            metrics = {
                'loss': results[0],
                'mae': results[1] if len(results) > 1 else np.nan
            }
        else:
            metrics = {'loss': results}

        y_pred = self._extract_predictions(self.model.predict(X_test))
        metrics['rmse'] = np.sqrt(np.mean((y_test - y_pred) ** 2))

        return metrics

    def _plot_metric(self, ax, history_dict: Dict, train_key: str, val_key: str, ylabel: str, title: str):
        ax.plot(history_dict[train_key], label='Train')
        ax.plot(history_dict[val_key], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    def plot_training_history(self, filename: str = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        history_dict = self.history.history

        loss_key = self._get_metric_key(history_dict, 'value_loss', 'loss')
        val_loss_key = self._get_metric_key(history_dict, 'val_value_loss', 'val_loss')
        mae_key = self._get_metric_key(history_dict, 'value_mae', 'mae')
        val_mae_key = self._get_metric_key(history_dict, 'val_value_mae', 'val_mae')

        self._plot_metric(ax1, history_dict, loss_key, val_loss_key, 'Loss (MSE)', 'Training and Validation Loss')
        self._plot_metric(ax2, history_dict, mae_key, val_mae_key, 'MAE', 'Training and Validation MAE')

        plt.tight_layout()

        if filename:
            plt.savefig(self.model_dir / filename)
        plt.show()

    def save_training_metrics(self, metrics: Dict, filename: str = "metrics.json"):
        save_path = self.model_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")
