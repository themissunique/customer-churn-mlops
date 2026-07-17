import os
from pathlib import Path

import joblib
import pandas as pd


def _train_missing_artifacts(base_dir: Path) -> None:
    current_dir = Path.cwd()
    os.chdir(base_dir)

    try:
        from src.pipeline.training_pipeline import TrainingPipeline

        TrainingPipeline().run_pipeline()
    finally:
        os.chdir(current_dir)


class PredictionPipeline:

    def __init__(self):

        base_dir = Path(__file__).resolve().parent.parent.parent
        preprocessor_path = base_dir / "artifacts" / "data_transformation" / "preprocessor.pkl"
        model_path = base_dir / "artifacts" / "model_trainer" / "model.pkl"

        if not preprocessor_path.exists() or not model_path.exists():
            _train_missing_artifacts(base_dir)

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)

    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        transformed = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed)

                
        return prediction[0]