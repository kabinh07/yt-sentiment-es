FROM python:3.10.12-slim-bullseye

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "update_sentiment.py"]