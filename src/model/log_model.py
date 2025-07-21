import pickle
import mlflow.pyfunc

class Mymodel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model_file"]  # 👈 dynamic path from MLflow
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)




