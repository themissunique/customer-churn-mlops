import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.pipeline.prediction_pipeline as prediction_module


def test_prediction_pipeline_trains_artifacts_when_missing(tmp_path, monkeypatch):
    module_path = tmp_path / "src" / "pipeline" / "prediction_pipeline.py"
    monkeypatch.setattr(prediction_module, "__file__", str(module_path))

    def fake_train(base_dir):
        preprocessor_path = base_dir / "artifacts" / "data_transformation" / "preprocessor.pkl"
        model_path = base_dir / "artifacts" / "model_trainer" / "model.pkl"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(Pipeline([("scaler", StandardScaler())]), preprocessor_path)
        joblib.dump(LogisticRegression(), model_path)

    monkeypatch.setattr(prediction_module, "_train_missing_artifacts", fake_train)

    pipeline = prediction_module.PredictionPipeline()

    assert pipeline.preprocessor is not None
    assert pipeline.model is not None
