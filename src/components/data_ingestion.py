import shutil

import pandas as pd

from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):

        self.config = config

    def initiate_data_ingestion(self):

        source_file = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

        shutil.copy(source_file, self.config.local_data_file)

        df = pd.read_csv(self.config.local_data_file)

        train_df, test_df = train_test_split(

            df,

            test_size=0.2,

            random_state=42,

            stratify=df["Churn"]

        )

        train_df.to_csv(

            self.config.train_data_path,

            index=False

        )

        test_df.to_csv(

            self.config.test_data_path,

            index=False

        )

        print("Data Ingestion Completed")