'''
This .py file is used to clean up the data
and move to bronze layer

Author: Vitor Abdo
Date: Oct/2024
'''

# import necessary packages
import sys
import os
import logging
import boto3
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# config
BUCKET_NAME_DATA = sys.argv[1]

logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def clean_data(BUCKET_NAME_DATA):
    '''
    Loads, processes, and normalizes CSV files stored in an S3 bucket, 
    separating the 2022 and 2023 data into distinct dataframes and renaming 
    columns according to a specific pattern.

    Parameters:
    -----------
    BUCKET_NAME_DATA : str
        Name of the S3 bucket where the CSV files are stored.

    Functionality:
    ---------------
    1. Creates an S3 client using boto3.
    2. Retrieves the list of objects in the S3 bucket and loads CSV files.
    3. For files containing '2022' in the name, the data is concatenated into the `df_hackers_2022` dataframe.
    4. For files containing '2023' in the name, the data is concatenated into the `df_hackers_2023` dataframe.
    5. Normalizes the columns of the 2022 dataframe, renaming them to a friendlier format.
    6. Normalizes the columns of the 2023 dataframe, if available.

    Returns:
    --------
    None
    '''
    # create a client instance for S3
    s3_client = boto3.client('s3')
    logging.info('S3 authentication was created successfully.')

    # Initialize dataframes
    df_hackers_2022 = pd.DataFrame()
    df_hackers_2023 = pd.DataFrame()

    # Get the list of objects from the S3 bucket
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME_DATA)

    # Loop through each file in the bucket
    for obj in response.get('Contents', []):
        key = obj['Key']

        if key.endswith('.csv'):
            logging.info(f'Reading {key} from S3 bucket.')

            # Load the CSV file from S3
            csv_obj = s3_client.get_object(Bucket=BUCKET_NAME_DATA, Key=key)
            current_df = pd.read_csv(csv_obj['Body'])

            # Check the file name to assign to the appropriate dataframe
            if '2022' in key:
                logging.info(f'Appending {key} to df_hackers_2022.')
                df_hackers_2022 = pd.concat(
                    [df_hackers_2022, current_df], ignore_index=True)
            elif '2023' in key:
                logging.info(f'Appending {key} to df_hackers_2023.')
                df_hackers_2023 = pd.concat(
                    [df_hackers_2023, current_df], ignore_index=True)

            logging.info(f'Dataset {key} loaded successfully.')

    logging.info('All datasets have been loaded.')

    # Normalize the 2022 dataframe
    df_hackers_2022 = df_hackers_2022[[
        "('P1_a_1 ', 'Faixa idade')",
        "('P1_b ', 'Genero')",
        "('P1_l ', 'Nivel de Ensino')",
        "('P1_m ', 'Área de Formação')",
        "('P2_h ', 'Faixa salarial')",
        "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')",
        "('P4_b_1 ', 'Dados relacionais (estruturados em bancos SQL)')",
        "('P4_b_2 ', 'Dados armazenados em bancos NoSQL')",
        "('P4_c_1 ', 'Dados relacionais (estruturados em bancos SQL)')",
        "('P4_c_2 ', 'Dados armazenados em bancos NoSQL')",
        "('P4_c_7 ', 'Planilhas')",
        "('P4_d_1 ', 'SQL')",
        "('P4_d_2 ', 'R ')",
        "('P4_d_3 ', 'Python')",
        "('P4_d_10 ', 'Scala')",
        "('P4_d_13 ', 'Javascript')",
        "('P4_d_14 ', 'Não utilizo nenhuma linguagem')",
        "('P6_a_1 ', 'Desenvolvo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')",
        "('P6_a_6 ', 'Desenvolvo/cuido da manutenção de repositórios de dados baseados em streaming de eventos como Data Lakes e Data Lakehouses.')",
        "('P6_c ', 'Sua organização possui um Data Lake?')",
        "('P6_h_1 ', 'Desenvolvendo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')",
        "('P7_a_1 ', 'Processo e analiso dados utilizando linguagens de programação como Python, R etc.')",
        "('P7_a_2 ', 'Realizo construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')",
        "('P7_a_3 ', 'Crio consultas através da linguagem SQL para exportar informações e compartilhar com as áreas de negócio.')",
        "('P7_d_2 ', 'Realizando construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')",
        "('P8_a_1 ', 'Estudos Ad-hoc com o objetivo de confirmar hipóteses, realizar modelos preditivos, forecasts, análise de cluster para resolver problemas pontuais e responder perguntas das áreas de negócio.')",
        "('P8_a_2 ', 'Sou responsável pela coleta e limpeza dos dados que uso para análise e modelagem.')",
        "('P8_a_4 ', 'Desenvolvo modelos de Machine Learning com o objetivo de colocar em produção em sistemas (produtos de dados).')",
        "('P8_a_5 ', 'Sou responsável por colocar modelos em produção, criar os pipelines de dados, APIs de consumo e monitoramento.')",
        "('P8_a_6 ', 'Cuido da manutenção de modelos de Machine Learning já em produção, atuando no monitoramento, ajustes e refatoração quando necessário.')",
        "('P8_a_10 ', 'Crio e gerencio soluções de Feature Store e cultura de MLOps.')",
        "('P8_a_11 ', 'Sou responsável por criar e manter a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')",
        "('P8_c_4 ', 'Ambientes de desenvolvimento na nuvem (Google Colab, AWS Sagemaker, Kaggle Notebooks etc)')",
        "('P8_c_7 ', 'Plataformas de Machine Learning (TensorFlow, Azure Machine Learning, Kubeflow etc)')",
        "('P8_c_9 ', 'Sistemas de controle de versão (Github, DVC, Neptune, Gitlab etc)')",
        "('P8_d_5 ', 'Colocando modelos em produção, criando os pipelines de dados, APIs de consumo e monitoramento.')",
        "('P8_d_10 ', 'Criando e gerenciando soluções de Feature Store e cultura de MLOps.')",
        "('P8_d_11 ', 'Criando e mantendo a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')",
        "('P2_f ', 'Cargo Atual')"
    ]]

    df_hackers_2022.rename(
        columns={
            "('P1_a_1 ', 'Faixa idade')": "faixa_idade",
            "('P1_b ', 'Genero')": "genero",
            "('P1_l ', 'Nivel de Ensino')": "nivel_ensino",
            "('P1_m ', 'Área de Formação')": "area_formacao",
            "('P2_h ', 'Faixa salarial')": "faixa_salarial",
            "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')": "tempo_experiencia_dados",
            "('P4_b_1 ', 'Dados relacionais (estruturados em bancos SQL)')": "experiencia_dados_relacionais_sql",
            "('P4_b_2 ', 'Dados armazenados em bancos NoSQL')": "experiencia_dados_no_sql",
            "('P4_c_1 ', 'Dados relacionais (estruturados em bancos SQL)')": "usa_dados_relacionais_sql",
            "('P4_c_2 ', 'Dados armazenados em bancos NoSQL')": "usa_dados_no_sql",
            "('P4_c_7 ', 'Planilhas')": "usa_planilhas",
            "('P4_d_1 ', 'SQL')": "usa_sql",
            "('P4_d_2 ', 'R ')": "usa_r",
            "('P4_d_3 ', 'Python')": "usa_python",
            "('P4_d_10 ', 'Scala')": "usa_scala",
            "('P4_d_13 ', 'Javascript')": "usa_javascript",
            "('P4_d_14 ', 'Não utilizo nenhuma linguagem')": "nao_usa_linguagem",
            "('P6_a_1 ', 'Desenvolvo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')": "desenvolve_pipelines_dados",
            "('P6_a_6 ', 'Desenvolvo/cuido da manutenção de repositórios de dados baseados em streaming de eventos como Data Lakes e Data Lakehouses.')": "desenvolve_repositorios_dados_datalakes",
            "('P6_c ', 'Sua organização possui um Data Lake?')": "empresa_possui_data_lake",
            "('P6_h_1 ', 'Desenvolvendo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')": "experiencia_pipelines_dados",
            "('P7_a_1 ', 'Processo e analiso dados utilizando linguagens de programação como Python, R etc.')": "desenvolve_analise_dados_programacao",
            "('P7_a_2 ', 'Realizo construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')": "desenvolve_dashboards_ferramentas_bi",
            "('P7_a_3 ', 'Crio consultas através da linguagem SQL para exportar informações e compartilhar com as áreas de negócio.')": "desenvolve_consultas_sql",
            "('P7_d_2 ', 'Realizando construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')": "experiencia_desenv_dashboards_ferramentas_bi",
            "('P8_a_1 ', 'Estudos Ad-hoc com o objetivo de confirmar hipóteses, realizar modelos preditivos, forecasts, análise de cluster para resolver problemas pontuais e responder perguntas das áreas de negócio.')": "desenvolve_estudos_modelos_preditivos",
            "('P8_a_2 ', 'Sou responsável pela coleta e limpeza dos dados que uso para análise e modelagem.')": "realiza_coleta_limpeza_para_analises",
            "('P8_a_4 ', 'Desenvolvo modelos de Machine Learning com o objetivo de colocar em produção em sistemas (produtos de dados).')": "desenvolve_modelos_machine_learning",
            "('P8_a_5 ', 'Sou responsável por colocar modelos em produção, criar os pipelines de dados, APIs de consumo e monitoramento.')": "colocar_modelos_machine_learning_producao",
            "('P8_a_6 ', 'Cuido da manutenção de modelos de Machine Learning já em produção, atuando no monitoramento, ajustes e refatoração quando necessário.')": "manutencao_modelos_machine_learning_producao",
            "('P8_a_10 ', 'Crio e gerencio soluções de Feature Store e cultura de MLOps.')": "cria_solucoes_feature_store_mlops",
            "('P8_a_11 ', 'Sou responsável por criar e manter a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')": "criar_manter_infraestrutura_modelos_e_solucoes_dados",
            "('P8_c_4 ', 'Ambientes de desenvolvimento na nuvem (Google Colab, AWS Sagemaker, Kaggle Notebooks etc)')": "usa_ambiente_desenvolvimento_nuvem",
            "('P8_c_7 ', 'Plataformas de Machine Learning (TensorFlow, Azure Machine Learning, Kubeflow etc)')": "usa_plataformas_ml_tensorflow_azureML_kubeflow_etc",
            "('P8_c_9 ', 'Sistemas de controle de versão (Github, DVC, Neptune, Gitlab etc)')": "usa_sistemas_controle_versao_github_dvc_etc",
            "('P8_d_5 ', 'Colocando modelos em produção, criando os pipelines de dados, APIs de consumo e monitoramento.')": "experiencia_modelos_machine_learning_producao",
            "('P8_d_10 ', 'Criando e gerenciando soluções de Feature Store e cultura de MLOps.')": "experiencia_solucoes_feature_store_mlops",
            "('P8_d_11 ', 'Criando e mantendo a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')": "experiencia_manter_infraestrutura_modelos_e_solucoes_dados",
            "('P2_f ', 'Cargo Atual')": "cargo_label"},
        inplace=True)
    logging.info('2022 dataframe was normalized successfully.')

    # Normalize the 2023 dataframe
    df_hackers_2023 = df_hackers_2023[[
        "('P1_a_1 ', 'Faixa idade')",
        "('P1_b ', 'Genero')",
        "('P1_l ', 'Nivel de Ensino')",
        "('P1_m ', 'Área de Formação')",
        "('P2_h ', 'Faixa salarial')",
        "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')",
        "('P4_b_1 ', 'Dados relacionais (estruturados em bancos SQL)')",
        "('P4_b_2 ', 'Dados armazenados em bancos NoSQL')",
        "('P4_c_1 ', 'Dados relacionais (estruturados em bancos SQL)')",
        "('P4_c_2 ', 'Dados armazenados em bancos NoSQL')",
        "('P4_c_7 ', 'Planilhas')",
        "('P4_d_1 ', 'SQL')",
        "('P4_d_2 ', 'R ')",
        "('P4_d_3 ', 'Python')",
        "('P4_d_10 ', 'Scala')",
        "('P4_d_14 ', 'JavaScript')",
        "('P4_d_15 ', 'Não utilizo nenhuma linguagem')",
        "('P4_j_1 ', 'Microsoft PowerBI')",
        "('P4_j_23 ', 'Não utilizo nenhuma ferramenta de BI no trabalho')",
        "('P6_a_1 ', 'Desenvolvo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')",
        "('P6_a_6 ', 'Desenvolvo/cuido da manutenção de repositórios de dados baseados em streaming de eventos como Data Lakes e Data Lakehouses.')",
        "('P6_b_21 ', 'Não utilizo ferramentas de ETL')",
        "('P6_c ', 'Sua organização possui um Data Lake?')",
        "('P6_h_1 ', 'Desenvolvendo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')",
        "('P7_a_1 ', 'Processo e analiso dados utilizando linguagens de programação como Python, R etc.')",
        "('P7_a_2 ', 'Realizo construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')",
        "('P7_a_3 ', 'Crio consultas através da linguagem SQL para exportar informações e compartilhar com as áreas de negócio.')",
        "('P7_d_2 ', 'Realizando construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')",
        "('P8_a_1 ', 'Estudos Ad-hoc com o objetivo de confirmar hipóteses, realizar modelos preditivos, forecasts, análise de cluster para resolver problemas pontuais e responder perguntas das áreas de negócio.')",
        "('P8_a_2 ', 'Sou responsável pela coleta e limpeza dos dados que uso para análise e modelagem.')",
        "('P8_a_4 ', 'Desenvolvo modelos de Machine Learning com o objetivo de colocar em produção em sistemas (produtos de dados).')",
        "('P8_a_5 ', 'Sou responsável por colocar modelos em produção, criar os pipelines de dados, APIs de consumo e monitoramento.')",
        "('P8_a_6 ', 'Cuido da manutenção de modelos de Machine Learning já em produção, atuando no monitoramento, ajustes e refatoração quando necessário.')",
        "('P8_a_10 ', 'Crio e gerencio soluções de Feature Store e cultura de MLOps.')",
        "('P8_a_11 ', 'Sou responsável por criar e manter a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')",
        "('P8_c_4 ', 'Ambientes de desenvolvimento na nuvem (Google Colab, AWS Sagemaker, Kaggle Notebooks etc)')",
        "('P8_c_7 ', 'Plataformas de Machine Learning (TensorFlow, Azure Machine Learning, Kubeflow etc)')",
        "('P8_c_9 ', 'Sistemas de controle de versão (Github, DVC, Neptune, Gitlab etc)')",
        "('P8_d_5 ', 'Colocando modelos em produção, criando os pipelines de dados, APIs de consumo e monitoramento.')",
        "('P8_d_10 ', 'Criando e gerenciando soluções de Feature Store e cultura de MLOps.')",
        "('P8_d_11 ', 'Criando e mantendo a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')",
        "('P2_f ', 'Cargo Atual')"
    ]]

    df_hackers_2023.rename(
        columns={
            "('P1_a_1 ', 'Faixa idade')": "faixa_idade",
            "('P1_b ', 'Genero')": "genero",
            "('P1_l ', 'Nivel de Ensino')": "nivel_ensino",
            "('P1_m ', 'Área de Formação')": "area_formacao",
            "('P2_h ', 'Faixa salarial')": "faixa_salarial",
            "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')": "tempo_experiencia_dados",
            "('P4_b_1 ', 'Dados relacionais (estruturados em bancos SQL)')": "experiencia_dados_relacionais_sql",
            "('P4_b_2 ', 'Dados armazenados em bancos NoSQL')": "experiencia_dados_no_sql",
            "('P4_c_1 ', 'Dados relacionais (estruturados em bancos SQL)')": "usa_dados_relacionais_sql",
            "('P4_c_2 ', 'Dados armazenados em bancos NoSQL')": "usa_dados_no_sql",
            "('P4_c_7 ', 'Planilhas')": "usa_planilhas",
            "('P4_d_1 ', 'SQL')": "usa_sql",
            "('P4_d_2 ', 'R ')": "usa_r",
            "('P4_d_3 ', 'Python')": "usa_python",
            "('P4_d_10 ', 'Scala')": "usa_scala",
            "('P4_d_14 ', 'JavaScript')": "usa_javascript",
            "('P4_d_15 ', 'Não utilizo nenhuma linguagem')": "nao_usa_linguagem",
            "('P4_j_1 ', 'Microsoft PowerBI')": "usa_powerbi",
            "('P4_j_23 ', 'Não utilizo nenhuma ferramenta de BI no trabalho')": "nao_usa_ferramentas_bi",
            "('P6_a_1 ', 'Desenvolvo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')": "desenvolve_pipelines_dados",
            "('P6_a_6 ', 'Desenvolvo/cuido da manutenção de repositórios de dados baseados em streaming de eventos como Data Lakes e Data Lakehouses.')": "desenvolve_repositorios_dados_datalakes",
            "('P6_b_21 ', 'Não utilizo ferramentas de ETL')": "nao_usa_ferramentas_etl",
            "('P6_c ', 'Sua organização possui um Data Lake?')": "empresa_possui_data_lake",
            "('P6_h_1 ', 'Desenvolvendo pipelines de dados utilizando linguagens de programação como Python, Scala, Java etc.')": "experiencia_pipelines_dados",
            "('P7_a_1 ', 'Processo e analiso dados utilizando linguagens de programação como Python, R etc.')": "desenvolve_analise_dados_programacao",
            "('P7_a_2 ', 'Realizo construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')": "desenvolve_dashboards_ferramentas_bi",
            "('P7_a_3 ', 'Crio consultas através da linguagem SQL para exportar informações e compartilhar com as áreas de negócio.')": "desenvolve_consultas_sql",
            "('P7_d_2 ', 'Realizando construções de dashboards em ferramentas de BI como PowerBI, Tableau, Looker, Qlik etc.')": "experiencia_desenv_dashboards_ferramentas_bi",
            "('P8_a_1 ', 'Estudos Ad-hoc com o objetivo de confirmar hipóteses, realizar modelos preditivos, forecasts, análise de cluster para resolver problemas pontuais e responder perguntas das áreas de negócio.')": "desenvolve_estudos_modelos_preditivos",
            "('P8_a_2 ', 'Sou responsável pela coleta e limpeza dos dados que uso para análise e modelagem.')": "realiza_coleta_limpeza_para_analises",
            "('P8_a_4 ', 'Desenvolvo modelos de Machine Learning com o objetivo de colocar em produção em sistemas (produtos de dados).')": "desenvolve_modelos_machine_learning",
            "('P8_a_5 ', 'Sou responsável por colocar modelos em produção, criar os pipelines de dados, APIs de consumo e monitoramento.')": "colocar_modelos_machine_learning_producao",
            "('P8_a_6 ', 'Cuido da manutenção de modelos de Machine Learning já em produção, atuando no monitoramento, ajustes e refatoração quando necessário.')": "manutencao_modelos_machine_learning_producao",
            "('P8_a_10 ', 'Crio e gerencio soluções de Feature Store e cultura de MLOps.')": "cria_solucoes_feature_store_mlops",
            "('P8_a_11 ', 'Sou responsável por criar e manter a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')": "criar_manter_infraestrutura_modelos_e_solucoes_dados",
            "('P8_c_4 ', 'Ambientes de desenvolvimento na nuvem (Google Colab, AWS Sagemaker, Kaggle Notebooks etc)')": "usa_ambiente_desenvolvimento_nuvem",
            "('P8_c_7 ', 'Plataformas de Machine Learning (TensorFlow, Azure Machine Learning, Kubeflow etc)')": "usa_plataformas_ml_tensorflow_azureML_kubeflow_etc",
            "('P8_c_9 ', 'Sistemas de controle de versão (Github, DVC, Neptune, Gitlab etc)')": "usa_sistemas_controle_versao_github_dvc_etc",
            "('P8_d_5 ', 'Colocando modelos em produção, criando os pipelines de dados, APIs de consumo e monitoramento.')": "experiencia_modelos_machine_learning_producao",
            "('P8_d_10 ', 'Criando e gerenciando soluções de Feature Store e cultura de MLOps.')": "experiencia_solucoes_feature_store_mlops",
            "('P8_d_11 ', 'Criando e mantendo a infra que meus modelos e soluções rodam (clusters, servidores, API, containers, etc.)')": "experiencia_manter_infraestrutura_modelos_e_solucoes_dados",
            "('P2_f ', 'Cargo Atual')": "cargo_label"},
        inplace=True)
    logging.info('2023 dataframe was normalized successfully.')

    # Concatenate all dataframes
    df_hackers_final = pd.concat(
        [df_hackers_2023, df_hackers_2022], ignore_index=True)
    df_hackers_final.dropna(subset=['cargo_label'], inplace=True)

    df_hackers_final.replace(
        {'cargo_label': {
            'Analista de Negócios/Business Analyst': 'Outra Opção',
            'Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas': 'Outra Opção',
            'Data Product Manager/ Product Manager (PM/APM/DPM/GPM/PO)': 'Outra Opção',
            'Analista de Suporte/Analista Técnico': 'Outra Opção',
            'Analista de Inteligência de Mercado/Market Intelligence': 'Outra Opção',
            'Outras Engenharias (não inclui dev)': 'Outra Opção',
            'Professor/Pesquisador': 'Outra Opção',
            'Estatístico': 'Outra Opção',
            'DBA/Administrador de Banco de Dados': 'Outra Opção',
            'Economista': 'Outra Opção',
            'Professor': 'Outra Opção',
            'Product Manager/ Product Owner (PM/APM/DPM/GPM/PO)': 'Outra Opção',
            'Engenheiro de Machine Learning/ML Engineer': 'Cientista de Dados/Data Scientist/Engenheiro de Machine Learning/ML Engineer/AI Engineer',
            'Analista de Marketing': 'Outra Opção',
            'Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect': 'Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect/Analytics Engineer',
            'Analytics Engineer': 'Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect/Analytics Engineer',
            'Cientista de Dados/Data Scientist': 'Cientista de Dados/Data Scientist/Engenheiro de Machine Learning/ML Engineer/AI Engineer',
            'Engenheiro de Machine Learning/ML Engineer/AI Engineer': 'Cientista de Dados/Data Scientist/Engenheiro de Machine Learning/ML Engineer/AI Engineer',
            'Analista de Dados/Data Analyst': 'Analista de Dados/Data Analyst/Analista de BI/BI Analyst',
            'Analista de BI/BI Analyst': 'Analista de Dados/Data Analyst/Analista de BI/BI Analyst'
        }
        }, inplace=True
    )

    df_hackers_final.replace(
        {'genero': {
            'Outro': 'Prefiro não informar'
        }
        }, inplace=True
    )

    df_hackers_final.replace(
        {'faixa_salarial': {
            'de R$ 101/mês a R$ 2.000/mês': 'de R$ 1.001/mês a R$ 2.000/mês'
        }
        }, inplace=True
    )

    df_hackers_final.replace(
        {'tempo_experiencia_dados': {
            'de 5 a 6 anos': 'de 4 a 6 anos'
        }
        }, inplace=True
    )

    df_hackers_final.replace(
        {'empresa_possui_data_lake': {
            0.0: 0,
            True: 1
        }
        }, inplace=True
    )

    encoder = LabelEncoder()
    df_hackers_final['cargo_label'] = encoder.fit_transform(
        df_hackers_final['cargo_label'])
    logging.info('All dataframes were normalized and cleaned successfully.')

    # Save the dataframe into bronze layer
    # Create the 'tmp' directory if it doesn't exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    df_hackers_final.to_csv(f'/tmp/data_hackers_final.csv', index=False)
    s3_client.upload_file(
        f'/tmp/data_hackers_final.csv',
        BUCKET_NAME_DATA,
        'bronze/data_hackers_final.csv')

    # Delete the temporary file
    os.remove(f'/tmp/data_hackers_final.csv')
    logging.info(f'Final dataframe was uploaded successfully.')


if __name__ == "__main__":
    logging.info('About to start executing the clean_data function...')
    clean_data(BUCKET_NAME_DATA)
    logging.info('Done executing the clean_data function.')
