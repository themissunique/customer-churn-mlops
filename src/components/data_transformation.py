import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):

        self.config = config


    def get_data_transformer_object(self):

        """
        Creates preprocessing pipeline
        """

        numerical_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]


        categorical_columns = [
            "gender",
            "Partner",
            "Dependents",
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
            "PaymentMethod"
        ]


        numerical_pipeline = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]
        )


        categorical_pipeline = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )

            ]
        )


        preprocessing = ColumnTransformer(
            transformers=[

                (
                    "numerical_pipeline",
                    numerical_pipeline,
                    numerical_columns
                ),

                (
                    "categorical_pipeline",
                    categorical_pipeline,
                    categorical_columns
                )

            ]
        )


        return preprocessing



    def initiate_data_transformation(self):

        try:

            train_path = (
                "artifacts/data_ingestion/train.csv"
            )

            test_path = (
                "artifacts/data_ingestion/test.csv"
            )


            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)


            print(
                "Train data loaded:",
                train_df.shape
            )

            print(
                "Test data loaded:",
                test_df.shape
            )


            # Remove unnecessary column

            train_df.drop(
                columns=["customerID"],
                inplace=True
            )


            test_df.drop(
                columns=["customerID"],
                inplace=True
            )


            # Convert TotalCharges

            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )


            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )


            target_column = "Churn"


            X_train = train_df.drop(
                columns=[target_column]
            )

            y_train = train_df[target_column]


            X_test = test_df.drop(
                columns=[target_column]
            )

            y_test = test_df[target_column]


            preprocessing_obj = (
                self.get_data_transformer_object()
            )


            X_train_transformed = (
                preprocessing_obj.fit_transform(
                    X_train
                )
            )


            X_test_transformed = (
                preprocessing_obj.transform(
                    X_test
                )
            )


            os.makedirs(
                os.path.dirname(
                    self.config.preprocessor_path
                ),
                exist_ok=True
            )


            joblib.dump(

                preprocessing_obj,

                self.config.preprocessor_path

            )


            print(
                "Preprocessor saved successfully"
            )


            return (

                X_train_transformed,

                X_test_transformed,

                y_train,

                y_test

            )


        except Exception as e:

            raise e