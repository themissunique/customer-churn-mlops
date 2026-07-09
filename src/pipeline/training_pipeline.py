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