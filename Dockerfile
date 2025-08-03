#base image
FROM python:3.9

# WORKDIR

WORKDIR /app

COPY flask_app/requirements.txt /app/requirements.txt

#copy
COPY flask_app/ /app/flask_app/

COPY models/vectorizer.pkl  /app/models/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn","-b","0.0.0.0:5000", "flask_app.app:app"]