import unittest
import mlflow
import os

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Nite2005"
        repo_name = "mlops-mini-project"

        # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        mlflow.set_tracking_uri("https://dagshub.com/Nite2005/mlops-mini-project.mlflow")


        cls.model_name = "emotion_detection"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'model:/{cls.model_name}/{cls.model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["staging"])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)



if __name__ == "__main__":
    unittest.main()



