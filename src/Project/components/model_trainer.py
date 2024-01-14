import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from Foodtimepredictor import logger
from Foodtimepredictor.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        logger.info("Model Trainer has been created")

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[self.config.target_column]
            test_y = test_data[self.config.target_column]

            # Ensure target variable is correctly formatted
            if train_y.ndim == 2:
                train_y = train_y.values.ravel()

            rf = RandomForestClassifier(n_estimators=self.config.n_estimators, min_samples_split=self.config.min_samples_split, random_state=42)
            rf.fit(train_x, train_y)

            # Save the model
            os.makedirs(self.config.root_dir, exist_ok=True)
            joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))
            logger.info(f"Model saved at {os.path.join(self.config.root_dir, self.config.model_name)}")

        except Exception as e:
            logger.exception(f"Error in training the model: {e}")
            raise e
