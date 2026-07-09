from types import SimpleNamespace

import numpy as np
from sklearn.datasets import make_classification

from src.components.model_trainer import ModelTrainer


def test_model_trainer_handles_string_labels(tmp_path):
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        random_state=42,
    )

    y = np.array(["No" if label == 0 else "Yes" for label in y])

    X_train = X[:100]
    X_test = X[100:]
    y_train = y[:100]
    y_test = y[100:]

    config = SimpleNamespace(
        root_dir=str(tmp_path / "models"),
        model_path=str(tmp_path / "models" / "best_model.joblib"),
    )

    trainer = ModelTrainer(config)
    score = trainer.train_models(X_train, y_train, X_test, y_test)

    assert score >= 0.0
