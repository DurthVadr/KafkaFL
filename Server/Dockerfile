FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install kafka-python tensorflow numpy scipy scikit-learn

COPY . .

CMD ["python", "server.py"]