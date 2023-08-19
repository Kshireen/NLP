FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir flask numpy scikit-learn scipy nltk joblib gunicorn 

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]