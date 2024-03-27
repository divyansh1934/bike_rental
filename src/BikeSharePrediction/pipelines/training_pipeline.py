
import os
import sys
from src.BikeSharePrediction.logger import logging
from src.BikeSharePrediction.exception.exception import customexception
import pandas as pd

from src.BikeSharePrediction.components.data_ingestion import DataIngestion
from src.BikeSharePrediction.components.data_transformation import DataTransformation
from src.BikeSharePrediction.components.model_trainer import ModelTrainer
from src.BikeSharePrediction.components.model_evaluation import ModelEvaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)
