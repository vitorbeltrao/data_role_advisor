'''
This file is for creating our inference api with fastapi

Author: Vitor Abdo
Date: Nov/2024
'''

# Import necessary packages
import json
import logging
import xgboost as xgb
import pickle
import os
import pandas as pd
import boto3
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


# Creating a Fastapi object
app = FastAPI()

class ModelInput(BaseModel):
    '''identifying the type of our model features'''
    experiencia_manter_infraestrutura_modelos_e_solucoes_dados: int
    desenvolve_pipelines_dados: int
    desenvolve_dashboards_ferramentas_bi: int
    experiencia_solucoes_feature_store_mlops: int
    nao_usa_linguagem: int
    usa_sql: int
    desenvolve_modelos_machine_learning: int
  
    class Config:
        schema_extra = {
            'example': {
                'experiencia_manter_infraestrutura_modelos_e_solucoes_dados': 0,
                'desenvolve_pipelines_dados': 0,
                'desenvolve_dashboards_ferramentas_bi': 1,
                'experiencia_solucoes_feature_store_mlops': 0,
                'nao_usa_linguagem': 0,
                'usa_sql': 1,
                'desenvolve_modelos_machine_learning': 0
            }
        }


# Function to load the model from S3
def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    model_path = '/tmp/model.xgb'
    try:
        s3.download_file(bucket_name, model_key, model_path)
        logging.info(f"Model downloaded to {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise

# Get mlflow model from s3 bucket
bucket_name = 'data-role-advisor-mlflow-artifacts' 
model_key = 'model.xgb' 
sk_pipe = load_model_from_s3(bucket_name, model_key)
logging.info('Get prod mlflow model: SUCCESS')


@app.get('/')
def greetings():
    '''get method to to greet a user'''
    return 'Welcome to our model API'


@app.post('/data_role_advisor_inference')
def data_role_advisor_pred(input_parameters: ModelInput):
    '''Post method for inference'''

    # Convert input data to a dictionary and then to a DataFrame
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    # Defina as colunas conforme esperado pelo modelo
    expected_columns = [
        'experiencia_manter_infraestrutura_modelos_e_solucoes_dados',
        'desenvolve_pipelines_dados',
        'desenvolve_dashboards_ferramentas_bi',
        'experiencia_solucoes_feature_store_mlops',
        'nao_usa_linguagem',
        'usa_sql',
        'desenvolve_modelos_machine_learning'
    ]

    # Construa o DataFrame com as colunas adequadas
    input_df = pd.DataFrame([input_dictionary], columns=expected_columns)

    # Converta o DataFrame para DMatrix para a previsão
    input_dmatrix = xgb.DMatrix(input_df)

    # Realize a previsão
    prediction = sk_pipe.predict(input_dmatrix)
    predicted_classes = prediction.argmax(axis=1)

    # Interprete o resultado da previsão
    if predicted_classes == 0:
        return 'You need a Data Analyst/BI analyst'
    elif predicted_classes == 1:
        return 'You need a Data Scientist/ML Engineer/AI Engineer'
    elif predicted_classes == 2:
        return 'You need a Data Engineer/Data Architect/Analytics Engineer'
    else:
        return 'You need another option like: Business Analyst, Developer, Software Engineer, Product Manager, Technical Analyst, Market Intelligence, DBA'


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)