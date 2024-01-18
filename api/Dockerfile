FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt app/requirements.txt
COPY api/main.py app/main.py
COPY models/financial_tweets_sentiment_model/ models/financial_tweets_sentiment_model/
RUN pip install --upgrade -r app/requirements.txt

# CMD exec uvicorn app.main:app --port 80 --host 0.0.0.0 --workers 1
CMD exec uvicorn app.main:app --port $PORT --host 0.0.0.0 --workers 1
