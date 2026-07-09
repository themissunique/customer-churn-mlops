from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path

    STATUS_FILE: Path

@dataclass(frozen=True)
class DataIngestionConfig:

    root_dir: Path

    local_data_file: Path

    train_data_path: Path

    test_data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:

    root_dir: Path

    preprocessor_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:

    root_dir: Path

    model_path: Path