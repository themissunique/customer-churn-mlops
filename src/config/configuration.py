from pathlib import Path

import yaml

from src.entity.config_entity import (DataIngestionConfig, DataTransformationConfig,
                                      DataValidationConfig, ModelTrainerConfig)


class ConfigurationManager:

    def __init__(self):

        with open("config.yaml") as f:

            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self):

        config = self.config["data_ingestion"]

        Path(config["root_dir"]).mkdir(parents=True, exist_ok=True)

        return DataIngestionConfig(

            root_dir=Path(config["root_dir"]),

            local_data_file=Path(config["local_data_file"]),

            train_data_path=Path(config["train_data_path"]),

            test_data_path=Path(config["test_data_path"])
        )
    
    def get_data_validation_config(self):

        config = self.config["data_validation"]

        Path(config["root_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        return DataValidationConfig(

            root_dir=Path(config["root_dir"]),

            STATUS_FILE=Path(config["STATUS_FILE"])

        )
    
    def get_data_transformation_config(self):

        config = self.config["data_transformation"]

        Path(config["root_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        return DataTransformationConfig(

            root_dir=Path(config["root_dir"]),

            preprocessor_path=Path(
                config["preprocessor_path"]
            )

        )
    
    def get_model_training_config(self):

        config = self.config["model_training"]

        Path(config["root_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )


        return ModelTrainerConfig(

            root_dir=Path(config["root_dir"]),

            model_path=Path(
                config["model_path"]
            )

        )