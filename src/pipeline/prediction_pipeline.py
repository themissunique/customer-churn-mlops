import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        self.preprocessor = joblib.load(
            "artifacts/data_transformation/preprocessor.pkl"
        )

        self.model = joblib.load(
            "artifacts/model_trainer/model.pkl"
        )


    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        transformed = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed)

        return prediction[0]