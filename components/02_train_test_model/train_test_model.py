'''
This .py file is used to train and test model
managed by mlflow tracking component

Author: Vitor Abdo
Date: Oct/2024
'''

# import necessary packages
import sys
import yaml
import logging
import mlflow
import timeit
import boto3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# config
BUCKET_NAME_DATA = sys.argv[1]
BUCKET_KEY_NAME = sys.argv[2]
all_feat = [
    'experiencia_manter_infraestrutura_modelos_e_solucoes_dados',
    'desenvolve_pipelines_dados',
    'desenvolve_dashboards_ferramentas_bi',
    'experiencia_solucoes_feature_store_mlops',
    'nao_usa_linguagem',
    'usa_sql',
    'desenvolve_modelos_machine_learning',
    'cargo_label']
binary_feat = [
    'experiencia_manter_infraestrutura_modelos_e_solucoes_dados',
    'desenvolve_pipelines_dados',
    'desenvolve_dashboards_ferramentas_bi',
    'experiencia_solucoes_feature_store_mlops',
    'nao_usa_linguagem',
    'usa_sql',
    'desenvolve_modelos_machine_learning']

ec2_client = boto3.client('ec2', region_name='us-east-1') 
response = ec2_client.describe_instances(
    Filters=[
        {'Name': 'instance-state-name', 'Values': ['running']}
    ]
)
active_instance_ids = [
    instance['InstanceId']
    for reservation in response['Reservations']
    for instance in reservation['Instances']
]

print("IDs das instâncias EC2 ativas:", active_instance_ids)


logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def get_inference_pipeline() -> Pipeline:
    '''
    Create and return a machine learning inference pipeline.

    This function constructs a pipeline for data preprocessing and model training.
    It preprocesses both qualitative and quantitative features, applies appropriate
    transformations, and sets up a RandomForest classifier with undersampling for handling
    class imbalance.

    Returns:
        tuple: A tuple containing the final pipeline and the list of processed feature names.
    '''
    binary_preproc = make_pipeline(
        SimpleImputer(fill_value=-1))

    preproc = ColumnTransformer(
        transformers=[
            ('binary', binary_preproc, binary_feat)
        ], remainder='passthrough')

    # combine all the processed feature names into a single list
    processed_features = binary_feat

    return preproc, processed_features


def feature_importance_plot(model, preprocessor, output_image_path) -> None:
    '''
    Generate and export a feature importance plot for a given model.

    Args:
        model: The trained model (e.g., RandomForest, XGBoost, LogisticRegression).
        preprocessor (ColumnTransformer): The preprocessor used in the pipeline.
        output_image_path (str): The path to save the output image.

    Returns:
        None
    '''

    # Try to get feature importances from different model types
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        raise ValueError(f"The model {type(model).__name__} does not support feature importances.")

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Create DataFrame with feature importances
    global_exp = pd.DataFrame(
        feature_importances,
        index=feature_names,
        columns=['importance'])

    # Sort the DataFrame by importance
    global_exp_sorted = global_exp.sort_values(
        by='importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(global_exp_sorted.index, global_exp_sorted['importance'])
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importance - {type(model).__name__}")
    plt.xticks(rotation=90)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_image_path)


def confusion_matrix_plot(y_true, y_pred, output_image_path) -> None:
    '''
    Generate and export a confusion matrix plot.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_image_path (str): The path to save the output image.
    '''
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Create a figure and axis
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Adjust the layout to prevent cutting off elements
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_image_path)


def load_model_configs(yaml_file):
    '''
    Load model configurations from a YAML file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed model configuration as a dictionary.
    '''
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def run_grid_search(
        config,
        experiment_name,
        dataset,
        test_size,
        label_column,
        cv,
        scoring,
        refit):
    '''
    Run a grid search, log parameters, metrics, and models to MLflow.

    Args:
        config (dict): Model configuration containing parameters and grid search settings.
        dataset (str): The path to the dataset file.
        test_size (float): The proportion of the dataset to include in the test split.
        label_column (str): The name of the column to use as the target variable.
        cv (int): The number of folds for cross-validation.
        scoring (list): List of scoring metrics for evaluation.
        refit (str): The metric to use for refitting the final model.
    '''
    date = datetime.today().strftime('%Y/%m/%d')

    train_set, test_set = train_test_split(
        dataset, test_size=test_size, random_state=42)

    preproc, _ = get_inference_pipeline()

    X_train = train_set.drop([label_column], axis=1)
    y_train = train_set[label_column]
    X_test = test_set.drop([label_column], axis=1)
    y_test = test_set[label_column]

    logging.info('Start tracking the model with mlflow...')

    # mlflow.set_tracking_uri('http://ec2-3-91-197-2.compute-1.amazonaws.com:5000')
    logging.info('Tracking server uri was connected.')

    # If experiment doesn't exist, create it
    if (not(mlflow.get_experiment_by_name(experiment_name))):
        mlflow.create_experiment(experiment_name)

    # Set up the running experiment to registry in mlflow
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = experiment.experiment_id

    # Log experiment metadata
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_param('Date', date)
    mlflow.log_param('Experiment_id', experiment_id)
    run = mlflow.active_run()
    mlflow.log_param('Active run', run.info.run_id)

    # Define pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preproc),
        (config['name'], eval(config['model']))
    ])

    # Define grid search
    logging.info('Training the model...')
    starttime = timeit.default_timer()
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config['param_grid'],
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        refit=refit
    )
    grid_search.fit(X_train, y_train)
    timing = timeit.default_timer() - starttime
    logging.info(f'The execution time of the model was:{timing}')

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    best_index = grid_search.best_index_
    for metric in scoring:
        best_train_score = grid_search.cv_results_[
            f'mean_train_{metric}'][best_index]
        best_val_score = grid_search.cv_results_[
            f'mean_test_{metric}'][best_index]
        mlflow.log_metric(f'train_{metric}', best_train_score)
        mlflow.log_metric(f'val_{metric}', best_val_score)

    # Log feature importances
    feature_importance_plot(
        grid_search.best_estimator_.named_steps[config['name']],
        grid_search.best_estimator_.named_steps['preprocessor'],
        f'feature_importance_{config['name']}.png')
    mlflow.log_artifact(f'feature_importance_{config['name']}.png')

    # Test on the test dataset
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test)
    confusion_matrix_plot(
        y_test,
        final_predictions,
        f'confusion_matrix_{config['name']}.png')
    for metric in scoring:
        if metric == 'accuracy' or metric == 'balanced_accuracy':
            test_score = balanced_accuracy_score(y_test, final_predictions)
        elif metric == 'precision':
            test_score = precision_score(y_test, final_predictions)
        elif metric == 'recall':
            test_score = recall_score(y_test, final_predictions)
        elif metric == 'f1':
            test_score = f1_score(y_test, final_predictions)
        else:
            raise ValueError(
                f'Metric {metric} is not implemented for logging on test data.')

        mlflow.log_metric(f'test_{metric}', test_score)
        mlflow.log_artifact(f'confusion_matrix_{config['name']}.png')

    # Log the model
    signature = infer_signature(X_test, final_predictions)
    if config['name'] == 'XGBClassifier':
        mlflow.xgboost.log_model(
            grid_search.best_estimator_.named_steps[config['name']],
            'best_model',
            signature=signature,
            input_example=X_test.head(1))
    else:
        mlflow.sklearn.log_model(
            grid_search.best_estimator_.named_steps[config['name']],
            'best_model',
            signature=signature,
            input_example=X_test.head(1))

    # Register the model
    mlflow.register_model(
        f'runs:/{run.info.run_id}/best_model',
        config['name'])

    mlflow.end_run()
    logging.info('Finish model tracking with mlflow.')


if __name__ == "__main__":
    logging.info('About to start executing the train_test_model function...')
    yaml_file = 'model_config.yaml'
    config = load_model_configs(yaml_file)

    # Get the dataset
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=BUCKET_NAME_DATA, Key=BUCKET_KEY_NAME)
    dataset = pd.read_csv(obj['Body'])
    cleaned_dataset = dataset[all_feat]

    # Execute the "run_grid_search" func
    for model_config in config['models']:
        run_grid_search(
            config=model_config,
            experiment_name=config['experiment']['name'],
            dataset=cleaned_dataset,
            test_size=config['experiment']['test_size'],
            label_column=config['experiment']['label_column'],
            cv=config['experiment']['cv'],
            scoring=config['experiment']['scoring'],
            refit=config['experiment']['refit'])

    logging.info('Done executing the train_test_model function.')
