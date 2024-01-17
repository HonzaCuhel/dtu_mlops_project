FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY main.py /code/main.py
RUN mkdir /code/models
COPY ../models/financial_tweets_sentiment_model /code/models/financial_tweets_sentiment_model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -t my_fastapi_app .
# docker run --name my_fastapi_app_container -p 80:80 my_fastapi_app