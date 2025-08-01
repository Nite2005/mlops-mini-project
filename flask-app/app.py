from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
#load model from model registry
import dagshub
import pickle


dagshub.init(repo_owner='Nite2005', repo_name='mlops-mini-project', mlflow=True)

# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Nite2005/mlops-mini-project.mlflow")





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

app.run(debug=True)