from Foodtimepredictor.config.configuration import ConfigurationManager
from Foodtimepredictor.components.data_validation import DataValiadtion
from Foodtimepredictor import logger


STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()
        
        # Validate all columns
        if data_validation.validate_all_columns():
            # If validation is successful, save the validated data
            data_validation.save_validated_data()
        else:
            logger.error("Data validation failed. Check the logs for details.")



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e