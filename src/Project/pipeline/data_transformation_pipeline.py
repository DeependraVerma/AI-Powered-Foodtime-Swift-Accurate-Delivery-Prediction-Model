from Foodtimepredictor.components.data_transformation import DataTransformation
from Foodtimepredictor import logger
from Foodtimepredictor.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        # Initialize the data transformation object and get preprocessing pipeline
        data_transformation.get_data_transformation_obj()

        # Splitting the data into training and test sets
        train_file_path, test_file_path = data_transformation.train_test_spliting()

        # Perform data transformation including feature engineering
        # and save processed objects and datasets
        data_transformation.inititate_data_transformation(train_file_path, test_file_path)

        logger.info("Data transformation completed successfully")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        pipeline = DataTransformationTrainingPipeline(config_manager)
        pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
