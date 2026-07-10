import os
from pyexpat import model
from xml.parsers.expat import model
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score


class ModelTrainer:


    def __init__(self, config):

        self.config = config



    def train_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Customer_Churn_Classification")

        models = {

            "LogisticRegression":
            LogisticRegression(
                max_iter=1000
            ),


            "RandomForest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        }



        best_model = None

        best_score = 0
        best_model_name = ""

        def _normalize_target(values):
            series = pd.Series(values).astype(str).str.strip()
            lowered = series.str.lower()

            if set(lowered.dropna().unique()) <= {"0", "1", "no", "yes"}:
                mapping = {"0": 0, "1": 1, "no": 0, "yes": 1}
                return lowered.map(mapping).astype(int)

            numeric_values = pd.to_numeric(series, errors="coerce")
            if numeric_values.notna().all():
                return numeric_values.astype(int)

            return pd.Series(
                pd.factorize(series)[0],
                index=series.index,
                name=series.name
            )

        y_train = _normalize_target(y_train)
        y_test = _normalize_target(y_test)

        for name, model in models.items():

            with mlflow.start_run(
                run_name=name
            ):

                print(
                    f"Training {name}"
                )


                model.fit(
                    X_train,
                    y_train
                )


                predictions = model.predict(
                    X_test
                )


                score = accuracy_score(
                    y_test,
                    predictions
                )
                mlflow.log_param(
                    "model_name",
                    name
                )
                mlflow.log_metric(
                    "accuracy",
                    score
                )
                
                mlflow.sklearn.log_model(

                    model,

                    "model",

                    registered_model_name="Customer_Churn_Model"

                )

                print(
                    name,
                    "Accuracy:",
                    score
                )


            if score > best_score:

                best_score = score

                best_model = model



        print(
            "Best Accuracy:",
            best_score
        )


        os.makedirs(

            self.config.root_dir,

            exist_ok=True

        )


        joblib.dump(

            best_model,

            self.config.model_path

        )


        print(
            "Best model saved"
        )


        return best_score