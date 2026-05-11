"""
Checkpoint management — save and load model weights, training states,
and experiment results.
"""

from config.paths import CHECKPOINT_DIR, CLASSICAL_MODELS, DL_MODELS
import pickle
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Manages model checkpoints and experiment states.
    """

    def __init__(self, experiment_name, checkpoint_dir=None):
        """
        Initialize checkpoint manager.

        Args:
            experiment_name: Name of the experiment
            checkpoint_dir: Directory for checkpoints (default: CHECKPOINT_DIR)
        """
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_classical_model(self, model, model_name, metadata=None):
        """
        Save a classical (sklearn) model.

        Args:
            model: Trained sklearn model
            model_name: Name for the saved model
            metadata: Optional dict with additional info

        Returns:
            Path: Path to saved model
        """
        save_path = CLASSICAL_MODELS / f"{model_name}.pkl"
        CLASSICAL_MODELS.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model': model,
            'model_name': model_name,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        with open(save_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"✓ Saved classical model: {save_path}")
        return save_path

    def load_classical_model(self, model_name):
        """
        Load a classical model.

        Args:
            model_name: Name of the model to load

        Returns:
            tuple: (model, metadata)
        """
        load_path = CLASSICAL_MODELS / f"{model_name}.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        with open(load_path, 'rb') as f:
            checkpoint = pickle.load(f)

        print(f"✓ Loaded classical model: {load_path}")
        return checkpoint['model'], checkpoint.get('metadata', {})

    def save_dl_model(self, model, model_name, optimizer=None,
                      epoch=None, history=None, metadata=None):
        """
        Save a deep learning (PyTorch/TensorFlow) model.

        Args:
            model: Trained model
            model_name: Name for the saved model
            optimizer: Optional optimizer state
            epoch: Current epoch number
            history: Training history
            metadata: Optional dict with additional info

        Returns:
            Path: Path to saved model
        """
        save_path = DL_MODELS / f"{model_name}.pth"
        DL_MODELS.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_name': model_name,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'history': history or {},
            'metadata': metadata or {}
        }

        # For PyTorch models
        if hasattr(model, 'state_dict'):
            checkpoint['model_state_dict'] = model.state_dict()
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # For Keras/TensorFlow models
        elif hasattr(model, 'save_weights'):
            weights_path = DL_MODELS / f"{model_name}_weights.h5"
            model.save_weights(str(weights_path))
            checkpoint['weights_path'] = str(weights_path)

        # Fallback: save entire model
        else:
            checkpoint['model'] = model

        torch.save(checkpoint, save_path)
        print(f"✓ Saved DL model: {save_path}")
        return save_path

    def load_dl_model(self, model_name, model_class=None):
        """
        Load a deep learning model.

        Args:
            model_name: Name of the model to load
            model_class: Model class for reconstruction (PyTorch)

        Returns:
            tuple: (model, metadata, history)
        """
        load_path = DL_MODELS / f"{model_name}.pth"

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        checkpoint = torch.load(load_path)

        # For PyTorch models
        if 'model_state_dict' in checkpoint and model_class is not None:
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])

        # For saved weights
        elif 'weights_path' in checkpoint:
            # Keras model should be built first
            raise NotImplementedError(
                "For Keras models, build the model first then call "
                "model.load_weights(checkpoint['weights_path'])"
            )

        # Fallback
        else:
            model = checkpoint.get('model')

        metadata = checkpoint.get('metadata', {})
        history = checkpoint.get('history', {})

        print(f"✓ Loaded DL model: {load_path}")
        return model, metadata, history

    def save_results(self, results, filename=None):
        """
        Save experiment results.

        Args:
            results: Dictionary of results
            filename: Optional filename (default: {experiment_name}_results.json)

        Returns:
            Path: Path to saved results
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.json"

        save_path = self.checkpoint_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Saved results: {save_path}")
        return save_path

    def load_results(self, filename=None):
        """
        Load experiment results.

        Args:
            filename: Optional filename (default: {experiment_name}_results.json)

        Returns:
            dict: Loaded results
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.json"

        load_path = self.checkpoint_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Results not found: {load_path}")

        with open(load_path, 'r') as f:
            results = json.load(f)

        print(f"✓ Loaded results: {load_path}")
        return results

    def checkpoint_exists(self, model_name, model_type='classical'):
        """
        Check if a checkpoint exists for a model.

        Args:
            model_name: Name of the model
            model_type: 'classical' or 'deep_learning'

        Returns:
            bool: True if checkpoint exists
        """
        if model_type == 'classical':
            path = CLASSICAL_MODELS / f"{model_name}.pkl"
        else:
            path = DL_MODELS / f"{model_name}.pth"

        return path.exists()

    def save_training_state(self, state, filename='training_state.pkl'):
        """
        Save complete training state for resumption.

        Args:
            state: Dictionary with training state
            filename: Filename for the state

        Returns:
            Path: Path to saved state
        """
        save_path = self.checkpoint_dir / filename

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"✓ Saved training state: {save_path}")
        return save_path

    def load_training_state(self, filename='training_state.pkl'):
        """
        Load training state to resume.

        Args:
            filename: Filename of the state

        Returns:
            dict: Training state
        """
        load_path = self.checkpoint_dir / filename

        if not load_path.exists():
            return None

        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        print(f"✓ Loaded training state: {load_path}")
        return state


def save_model_simple(model, filepath):
    """
    Simple helper to save any model.

    Args:
        model: Model to save
        filepath: Path to save to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ Model saved: {filepath}")


def load_model_simple(filepath):
    """
    Simple helper to load any model.

    Args:
        filepath: Path to load from

    Returns:
        Loaded model
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"✓ Model loaded: {filepath}")
    return model


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    # Test checkpoint manager
    manager = CheckpointManager('test_experiment')

    # Create and save a dummy classical model
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save model
    manager.save_classical_model(
        model,
        'test_rf',
        metadata={'accuracy': 0.95, 'n_features': 10}
    )

    # Load model
    loaded_model, metadata = manager.load_classical_model('test_rf')
    print(f"Loaded model metadata: {metadata}")

    # Save results
    results = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'predictions': np.array([0, 1, 1, 0])
    }
    manager.save_results(results)

    # Load results
    loaded_results = manager.load_results()
    print(f"Loaded results: {loaded_results}")

    print("\n✓ Checkpoint manager test complete!")
