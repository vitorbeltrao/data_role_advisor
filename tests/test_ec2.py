import pytest
import requests
import socket
import mlflow
from urllib.parse import urlparse

class TestMLflowConnection:
    
    @pytest.fixture
    def mlflow_tracking_uri(self):
        """Fixture que retorna a URI do MLflow tracking server"""
        return 'http://ec2-3-91-197-2.compute-1.amazonaws.com:5000'
    
    @pytest.fixture
    def parsed_uri(self, mlflow_tracking_uri):
        """Fixture que faz o parse da URI do MLflow"""
        return urlparse(mlflow_tracking_uri)
    
    def test_can_resolve_hostname(self, parsed_uri):
        """Testa se o hostname pode ser resolvido para um IP"""
        try:
            socket.gethostbyname(parsed_uri.hostname)
            assert True
        except socket.gaierror as e:
            pytest.fail(f"Não foi possível resolver o hostname: {e}")
    
    def test_port_is_open(self, parsed_uri):
        """Testa se a porta está aberta e acessível"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((parsed_uri.hostname, parsed_uri.port))
            assert result == 0, f"A porta {parsed_uri.port} não está acessível"
        finally:
            sock.close()
    
    def test_mlflow_api_health(self, mlflow_tracking_uri):
        """Testa se a API do MLflow está respondendo"""
        try:
            response = requests.get(f"{mlflow_tracking_uri}/health")
            assert response.status_code == 200, "API do MLflow não está saudável"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Falha ao conectar com a API do MLflow: {e}")
    
    def test_can_set_tracking_uri(self, mlflow_tracking_uri):
        """Testa se é possível configurar a tracking URI no MLflow"""
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            current_uri = mlflow.get_tracking_uri()
            assert current_uri == mlflow_tracking_uri
        except Exception as e:
            pytest.fail(f"Falha ao configurar tracking URI: {e}")
    
    def test_can_list_experiments(self, mlflow_tracking_uri):
        """Testa se é possível listar experimentos"""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        try:
            experiments = mlflow.search_experiments()
            assert isinstance(experiments, list)
        except Exception as e:
            pytest.fail(f"Falha ao listar experimentos: {e}")
    
    def test_can_create_and_delete_experiment(self, mlflow_tracking_uri):
        """Testa se é possível criar e deletar um experimento"""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        test_exp_name = "test_experiment_connection"
        
        try:
            # Tenta deletar o experimento se já existir
            existing_exp = mlflow.get_experiment_by_name(test_exp_name)
            if existing_exp:
                mlflow.delete_experiment(existing_exp.experiment_id)
            
            # Cria novo experimento
            experiment_id = mlflow.create_experiment(test_exp_name)
            assert experiment_id is not None
            
            # Deleta o experimento
            mlflow.delete_experiment(experiment_id)
            assert True
            
        except Exception as e:
            pytest.fail(f"Falha ao criar/deletar experimento: {e}")
