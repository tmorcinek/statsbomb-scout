"""Module for training and evaluating the model."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt

import config


class ModelTrainer:
    """Handles model training, evaluation, and saving."""

    def __init__(self, model: keras.Model, model_dir: Path = Path(f"models/{config.MODEL_TYPE}/")):
        """
        Initialize trainer.

        Args:
            model: Keras model to train
            model_dir: Directory to save models
        """
        self.model = model
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.history = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             batch_size: int = 32, epochs: int = 50) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs

        Returns:
            Training history
        """
        # Define callbacks
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
                filepath=str(self.model_dir / "best_model.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'test_loss': results[0],
            'test_mae': results[1]
        }

        # Additional metrics
        y_pred = self.model.predict(X_test)
        metrics['rmse'] = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))

        print(f"\nTest Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics

    def plot_training_history(self, filename: str = None):
        """
        Plot training history.

        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE plot
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Val MAE')
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
        """
        Save training metrics to JSON.

        Args:
            metrics: Dictionary of metrics
            filename: Filename for metrics
        """
        save_path = self.model_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")

