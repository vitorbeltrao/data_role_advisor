'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import argparse
import mlflow

# define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=str, default='all', help='Steps to execute')

_steps = [
    # 'basic_clean',
    # 'data_check',
    # 'train_model',
    # 'test_model',
    # 'deployment'
]

def main():
    '''
    Executes the end-to-end pipeline using MLflow based on the specified steps.

    This function reads command line arguments to determine which steps of the pipeline 
    should be executed. The steps can include 'basic_clean', 'data_check', 'train_model', 
    'test_model', and 'deployment'. By default, all steps are run if no specific steps 
    are provided or if 'all' is passed as a parameter.

    The pipeline steps are executed via MLflow projects using their corresponding URIs.

    Args:
        steps (str): A comma-separated string specifying which steps to execute. 
                     The default value is 'all', which runs the entire pipeline. 
                     Valid options include 'basic_clean', 'data_check', 'train_model', 
                     'test_model', 'deployment'.

    Example:
        To run only the 'basic_clean' and 'train_model' steps:
        ```
        python main.py --steps basic_clean,train_model
        ```

        To run all steps:
        ```
        python main.py --steps all
        ```
    '''
    # read command line arguments
    args = parser.parse_args()

    # Steps to execute
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    if 'basic_clean' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/data_role_advisor#components/01_basic_clean'
        mlflow.run(project_uri, parameters={'steps': 'basic_clean'})

    # if 'data_check' in active_steps:
    #     project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/04_data_check'
    #     mlflow.run(project_uri)

    # if 'train_model' in active_steps:
    #     project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/05_train_model'
    #     mlflow.run(project_uri, parameters={'steps': 'train_model'})

    # if 'test_model' in active_steps:
    #     project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/06_test_model'
    #     mlflow.run(project_uri, parameters={'steps': 'test_model'})

    # if 'deployment' in active_steps:
    #     project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/07_deployment'
    #     mlflow.run(project_uri, parameters={'steps': 'deployment'})

if __name__ == "__main__":
    main()