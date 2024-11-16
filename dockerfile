FROM continuumio/miniconda3:latest

# Definir o diretório de trabalho
WORKDIR /train_app

# Copiar os arquivos para o contêiner
COPY . /train_app

# Argumentos para receber valores no momento da construção do contêiner
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

# Configurar variáveis de ambiente a partir dos argumentos
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=$AWS_REGION

# Instalar os pacotes necessários
RUN conda env create -f environment.yaml

# Ativar o ambiente Conda no ENTRYPOINT e rodar o script
ENTRYPOINT ["bash", "-c", 'source activate data_role_advisor && mlflow run . --experiment-name="Data Role Advisor Experiment I"']