from pathlib import Path

import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        preprocessor_path = BASE_DIR / "artifacts" / "data_transformation" / "preprocessor.pkl"

        self.preprocessor = joblib.load(preprocessor_path)

        self.model = joblib.load(
            BASE_DIR / "artifacts" / "model_trainer" / "model.pkl"
        )


    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        transformed = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed)

        return prediction[0]