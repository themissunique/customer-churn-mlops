import pandas as pd

from src.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self, config):

        self.config = config

    def validate(self):

        train = pd.read_csv(
            "artifacts/data_ingestion/train.csv"
        )

        required_columns = [

            "customerID",

            "gender",

            "SeniorCitizen",

            "Partner",

            "Dependents",

            "tenure",

            "PhoneService",

            "MultipleLines",

            "InternetService",

            "OnlineSecurity",

            "OnlineBackup",

            "DeviceProtection",

            "TechSupport",

            "StreamingTV",

            "StreamingMovies",

            "Contract",

            "PaperlessBilling",

            "PaymentMethod",

            "MonthlyCharges",

            "TotalCharges",

            "Churn"

        ]

        validation_status = all(

            column in train.columns

            for column in required_columns

        )

        with open(

            self.config.STATUS_FILE,

            "w"

        ) as f:

            f.write(

                f"Validation status: {validation_status}"

            )

        return validation_status