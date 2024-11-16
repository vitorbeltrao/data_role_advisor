'''
Main script to execute the machine learning pipeline using MLflow Projects.

This file orchestrates the necessary steps to perform an end-to-end machine learning 
workflow. It uses MLflow Projects to manage and execute each stage, such as data 
cleaning, training, testing, and optional deployment.

Author: Vitor Abdo
Date: October 2024
'''

import argparse
import mlflow

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=str, default='all', help='Pipeline steps to execute')

# Define the list of steps in the pipeline
_steps = [
    'basic_clean',
    'train_test_model'
]

def main():
    '''
    Runs specified steps in the ML pipeline, controlled through command-line arguments.

    This function processes the steps argument to determine which pipeline stages to execute. 
    Available steps include 'basic_clean' for data preprocessing and 'train_test_model' for 
    training and testing the model. By default, if no specific steps are provided or 'all' 
    is passed, the entire pipeline is executed.

    The steps are launched via MLflow Projects, which directs the MLflow tracking server 
    and fetches project components using designated URIs.

    Args:
        --steps (str): A comma-separated list indicating steps to execute, e.g., 
                       'basic_clean,train_test_model'. Default is 'all' to run every step.
    '''
    # Parse command-line arguments
    args = parser.parse_args()

    # Determine which steps to run
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    # Run data cleaning step if specified
    if 'basic_clean' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/data_role_advisor#components/01_basic_clean'
        mlflow.run(project_uri, parameters={'steps': 'basic_clean'})

    # Run training and testing step if specified
    if 'train_test_model' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/data_role_advisor#components/02_train_test_model'
        mlflow.run(project_uri)


if __name__ == "__main__":
    main()
