import os
from Foodtimepredictor import logger
import pandas as pd
from Foodtimepredictor.config.configuration import DataValidationConfig

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
        
    def save_validated_data(self):
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            # Perform any additional validation or transformation if needed
            # ...

            # Save the validated data to finalTrain.csv
            validated_data_path = os.path.join(self.config.root_dir, 'finalTrain.csv')
            data.to_csv(validated_data_path, index=False)
            logger.info(f"Validated data saved at {validated_data_path}")
        except Exception as e:
            logger.error(f"Error in saving validated data: {e}")
            raise e