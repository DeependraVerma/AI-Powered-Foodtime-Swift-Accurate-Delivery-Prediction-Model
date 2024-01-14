import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Foodtimepredictor.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from Foodtimepredictor.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from Foodtimepredictor import logger
from sklearn.metrics import accuracy_score,confusion_matrix


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        logger.info(f"Model accuracy_score is: {acc}")
        return acc

    def save_results(self):
        # Load the preprocessed test data
        preprocessed_test_data = pd.read_csv('artifacts/data_transformation/processed_test.csv')

        # Separate features and target variable
        test_x = preprocessed_test_data.drop([self.config.target_column], axis=1)
        test_y = preprocessed_test_data[self.config.target_column]

        # Load the trained model
        model = joblib.load(self.config.model_path)

        # Make predictions and evaluate
        predicted_qualities = model.predict(test_x)
        acc = self.eval_metrics(test_y, predicted_qualities)

        # Saving metrics as local
        scores = {"accuracy": acc}
        save_json(path=Path(self.config.metric_file_name), data=scores)

