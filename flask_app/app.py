from flask import Flask,render_template,request
import mlflow
from flask_app.preprocessing_utility import normalize_text
#load model from model registry

import dagshub
import pickle
import os


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "Nite2005"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


app = Flask(__name__)
model_name = "emotion_detection"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"


model = mlflow.pyfunc.load_model(model_uri)



@app.route('/')
def home():
    return render_template("index.html",result=None)


@app.route('/predict',methods=['POST'])
def predict():

    text = request.form['text']   
     #clean
    text = normalize_text(text)
 
    #bow
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)


    features = vectorizer.transform([text])
    
     #prediction

    result =  model.predict(features)

    return render_template('index.html',result=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)