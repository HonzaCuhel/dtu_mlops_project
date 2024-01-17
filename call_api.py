import requests

url = "https://deployed-financial-tweet-sentiment-o64hln5vbq-ew.a.run.app/predict_batch/"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
data = [
    "I think $TSLA is going to the moon!"
]

response = requests.post(url, headers=headers, json=data)
print(response.json())
