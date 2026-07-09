from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

class TrainingPipeline:

    def __init__(self):

        pass

    def run_pipeline(self):

        config = ConfigurationManager()

        ingestion_config = config.get_data_ingestion_config()

        ingestion = DataIngestion(

            ingestion_config

        )

        ingestion.initiate_data_ingestion()
        
        validation_config = config.get_data_validation_config()

        validation = DataValidation(
            validation_config
        )

        status = validation.validate()

        print(status)

        transformation_config = (
            config.get_data_transformation_config()
        )


        data_transformation = DataTransformation(
            transformation_config
        )


        X_train, X_test, y_train, y_test = (
            data_transformation.initiate_data_transformation()
        )

        model_config = (
            config.get_model_training_config()
        )


        model_trainer = ModelTrainer(
            model_config
        )


        model_trainer.train_models(

            X_train,

            y_train,

            X_test,

            y_test

        )