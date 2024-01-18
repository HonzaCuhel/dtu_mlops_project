import streamlit as st
import requests

def main():
    user_input = st.text_area("Enter text:", "")

    if st.button("Predict"):
        ml_container_url = "https://deployed-financial-tweet-sentiment-o64hln5vbq-ew.a.run.app/predict_batch/"  

        payload = [user_input]
        header = {"Content-Type": "application/json",}
        response = requests.post(ml_container_url, headers=header, json=payload)

        if response.status_code == 200:
            prediction = response.json().get("output", ["N/A"])[0]
            st.write("Prediction:", prediction)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
