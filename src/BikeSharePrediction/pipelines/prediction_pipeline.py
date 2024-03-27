import os
import sys
import pandas as pd
from src.BikeSharePrediction.exception.exception import customexception
from src.BikeSharePrediction.logger import logging
from src.BikeSharePrediction.utils.utils import load_object


class PredictPipeline:

    
    def __init__(self):
        print("init.. the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise customexception(e,sys)


class CustomData:
    def __init__(self,
                 season: int, 
                 yr: int, 
                 mnth: int,   
                 holiday: int,
                 weekday : int,
                 workingday :int,
                 weathersit: int, 
                 temp:float,
                 atemp:float,
                 hum: float,
                 windspeed :float):
        
        self.season=season
        self.yr=yr
        self.mnth=mnth
        self.holiday=holiday
        self.weekday=weekday
        self.workingday=workingday
        self.weathersit=weathersit
        self.temp=temp
        self.atemp=atemp
        self.hum=hum
        self.windspeed=windspeed
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'season':[self.season],
                'yr':[self.yr],
                'mnth':[self.mnth],
                'holiday':[self.holiday],
                'weekday':[self.weekday],
                'workingday':[self.workingday],
                'weathersit':[self.weathersit],
                'temp':[self.temp],
                'atemp':[self.atemp],
                'hum':[self.hum],
                'windspeed':[self.windspeed]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)