import argparse
import pandas as pd
import numpy as np
import datetime
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from joblib import dump, load

import data_processor
import model_registry

logging.basicConfig(level=logging.INFO)

features = ['Age','KM','FuelType','HP','MetColor','Automatic','CC','Doors','Weight']
categorical_features = ['FuelType']
label = 'Price'

def run(data_path, model_path, r2_criteria):
    logging.info('Process Data...')
    df = data_processor.run(data_path)
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ], 
        remainder='passthrough'
    )
    
    #Train-Test Split
    logging.info('Start Train-Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(df[features], \
                                                        df[label], \
                                                        test_size=0.2, \
                                                        random_state=0)
    
    #Train Classifier
    logging.info('Start Training...')
    lr = LinearRegression()
    
    regr = Pipeline(steps=[("preprocessor", preprocessor),\
                          ("regression", lr)
                         ])
    regr.fit(X_train, y_train)
    
    #Evaluate and Deploy
    logging.info('Evaluate...')
    r2_score = regr.score(X_test, y_test)
    if  r2_score >= r2_criteria:
        logging.info('Deploy...')
        mdl_meta = { 'name': 'used_car_price_prediction', 'metrics': f"r2_score:{r2_score}" }
        model_registry.register(regr, features, mdl_meta)
    
    logging.info('Training completed.')
    return None

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--r2_criteria", type=float)
    args = argparser.parse_args()
    run(args.data_path, args.model_path, args.r2_criteria)