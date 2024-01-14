import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from Foodtimepredictor.entity.config_entity import DataTransformationConfig
from Foodtimepredictor.utils.common import save_obj
from Foodtimepredictor import logger

class Feature_Engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logger.info("****************** Feature Engineering started ******************")

    def distance_numpy(self, df, lat1, lon1, lat2, lon2):
        p = np.pi / 180
        a = 0.5 - np.cos((df[lat2] - df[lat1]) * p)/2 + np.cos(df[lat1] * p) * np.cos(df[lat2] * p) * (1 - np.cos((df[lon2] - df[lon1]) * p))/2
        df['distance'] = 12734 * np.arcsin(np.sqrt(a))

    def transform_data(self, df):
        try:
            df.drop(['ID'], axis=1, inplace=True)
            self.distance_numpy(df, 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude')
            df.drop(['Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1, inplace=True)
            logger.info("Dropping columns from our original dataset")
            return df
        except Exception as e:
            raise e

    def fit(self, X, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        try:    
            transformed_df = self.transform_data(X)
            return transformed_df
        except Exception as e:
            raise e

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config

    def get_data_transformation_obj(self):
        try:
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']
            categorical_columns = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'Weather_conditions', 'Road_traffic_density']
            numerical_column = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 'multiple_deliveries', 'distance']

            numerical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            categorical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            ordinal_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density, Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])

            self.preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_column),
                ('categorical_pipeline', categorical_pipeline, categorical_columns)
            ], remainder='passthrough')

            logger.info("Pipeline steps completed")
            return             self.preprocessor
        except Exception as e:
            logger.exception(f"Error in get_data_transformation_obj: {e}")
            raise e

    def transform_and_save_data(self, df, file_path):
        try:
            # Apply the preprocessing pipeline
            processed_data = self.preprocessor.fit_transform(df)
            new_column_names = self.preprocessor.get_feature_names_out()

            # Mapping new column names to original names
            renamed_columns = [name.split('__')[1] if '__' in name else name for name in new_column_names]
            processed_df = pd.DataFrame(processed_data, columns=renamed_columns)

            # Save the processed dataframe
            processed_df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.exception(f"Error in transform_and_save_data: {e}")
            raise e

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[("fe", Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise e

    def train_test_spliting(self):
        data = pd.read_csv(self.data_transformation_config.data_path)
        train, test = train_test_split(data, random_state=42, test_size=0.2)
        train_file_path = os.path.join(self.data_transformation_config.root_dir, "train.csv")
        test_file_path = os.path.join(self.data_transformation_config.root_dir, "test.csv")
        train.to_csv(train_file_path, index=False)
        test.to_csv(test_file_path, index=False)
        logger.info("Splited data into training and test sets")
        return train_file_path, test_file_path

    def inititate_data_transformation(self, train_file_path, test_file_path):
        try:
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            fe_obj = self.get_feature_engineering_object()
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)
            self.transform_and_save_data(train_df, self.data_transformation_config.transformed_train_path)
            self.transform_and_save_data(test_df, self.data_transformation_config.transformed_test_path)
            save_obj(obj=self.preprocessor, file_path=self.data_transformation_config.preprocessed_obj_path)
            save_obj(obj=fe_obj, file_path=self.data_transformation_config.feature_engg_obj_path)
        except Exception as e:
            logger.exception(f"Error in inititate_data_transformation: {e}")
            raise e

