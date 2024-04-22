import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
from apify_client import ApifyClient
#loading model
model = joblib.load("classifier.pkl")
# Initialize the ApifyClient with your API token
client = ApifyClient("apify_api_nscRkHOyMh3mytIWftXpHpZlIzBhgF4mZyPV")

# Prepare the Actor input
st.title("Fake Instagram Profile Detection")
st.write("Plaese provide instagram account details you would like to predict")
n = st.text_input("Enter username ")
run_input = { "usernames": [n] }

# Run the Actor and wait for it to finish
run = client.actor("dSCLg0C3YEZ83HzYX").call(run_input=run_input)

m = client.dataset(run["defaultDatasetId"])
# Fetch and print Actor results from the run's dataset (if there are any)
for item in m.iterate_items():
        postsCount= item.get('postsCount')
        followersCount = item.get('followersCount')
        followsCount = item.get('followsCount')
        private=item.get('private')
        verified=item.get('verified')


def predictor(postsCount,followersCount,followsCount,private,verified):
    prediction = model.predict([[postsCount,followersCount,followsCount,private,verified]])
    print(prediction)
    return prediction




if st.button("Predict"):
    result = predictor(postsCount,followersCount,followsCount,private,verified)
    st.write("The number of posts : " , postsCount)
    st.write("The number of followers : " ,followersCount)
    st.write("The number of following : " ,followsCount)
    st.write("Private : " ,private)
    st.write("Verified : " ,verified)
    if result==0:
        st.error("The Account is Likely to be Fake ")
    else:
        st.success("The Account is Likely to be Real")
