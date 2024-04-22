import numpy as np
import pandas as pd
import streamlit as st
import joblib
from apify_client import ApifyClient
model = joblib.load("FPD/classifier.pkl")
client = ApifyClient("apify_api_nscRkHOyMh3mytIWftXpHpZlIzBhgF4mZyPV")
st.title("Fake Instagram Profile Detection")
st.write("Please provide instagram account details you would like to predict")
n = st.text_input("Enter username ")
run_input = { "usernames": [n] }
run = client.actor("dSCLg0C3YEZ83HzYX").call(run_input=run_input)
m = client.dataset(run["defaultDatasetId"])
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
        st.success("The Account is Likely to be Fake ")
    else:
        st.error("The Account is Likely to be Real")
