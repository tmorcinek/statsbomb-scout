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

        self.has_multiple_outputs = isinstance(model.output, dict) or (isinstance(model.output, list) and len(model.output) > 1)

        self.output_names = self.model.output_names if hasattr(self.model, 'output_names') else []

    def _prepare_labels(self, y: Union[np.ndarray, dict], X_shape: int) -> Union[Dict, np.ndarray]:
        if not self.has_multiple_outputs:
            return y

        labels = {self.VALUE_OUTPUT: y}
        if self.ATTENTION_OUTPUT in self.output_names:
            # Provide dummy targets for attention weights (loss weight is 0, so values don't matter)
            labels[self.ATTENTION_OUTPUT] = np.zeros((len(y), X_shape))

        return labels

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


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        # Prepare labels for multi-output models
        y_test_prepared = self._prepare_labels(y_test, X_test.shape[1])

        results = self.model.evaluate(X_test, y_test_prepared, verbose=0)

        # Handle different result formats
        if isinstance(results, list):
            metrics = {
                'test_loss': results[0],
                'test_mae': results[1] if len(results) > 1 else results[0]
            }
        else:
            metrics = {'test_loss': results}

        # Additional metrics - extract value predictions
        predictions = self.model.predict(X_test)
        if isinstance(predictions, dict):
            y_pred = predictions[self.VALUE_OUTPUT].flatten()
        else:
            y_pred = predictions.flatten()

        metrics['test_rmse'] = np.sqrt(np.mean((y_test - y_pred) ** 2))

        print(f"\nTest Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics

    def plot_training_history(self, filename: str = None):
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Determine metric keys (handle both single and multi-output models)
        history_dict = self.history.history
        loss_key = next((k for k in ['value_loss', 'loss'] if k in history_dict), 'loss')
        val_loss_key = next((k for k in ['val_value_loss', 'val_loss'] if k in history_dict), 'val_loss')
        mae_key = next((k for k in ['value_mae', 'mae'] if k in history_dict), 'mae')
        val_mae_key = next((k for k in ['val_value_mae', 'val_mae'] if k in history_dict), 'val_mae')

        # Loss plot
        ax1.plot(history_dict[loss_key], label='Train Loss')
        ax1.plot(history_dict[val_loss_key], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE plot
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
