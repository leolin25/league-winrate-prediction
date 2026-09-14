import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd

# page configurations
st.title('League of Legends: 10-Minute Win Predictor')

st.sidebar.header('Match Context')
st.warning('**Disclaimer:** This model is trained on Season 10 data from 2020. The current meta, map objectives, and champion baselines have changed significantly.')

# load the model
model = joblib.load('models/baseline.joblib')

# sidebar for user to fill in stats
st.sidebar.header('Match State at 10 mins (Blue)')

# continuous differentials
gold_diff = st.sidebar.slider('Gold Differential', min_value=-5000, max_value=5000, value=0, step=100)
xp_diff = st.sidebar.slider('XP Differential', min_value=-5000, max_value=5000, value=0, step=100)
cs_diff = st.sidebar.slider('CS Differential', min_value=-50, max_value=50, value=0, step=1)

# discrete differentials
st.sidebar.subheader('Objectives')
blue_dragons = st.sidebar.number_input('Blue Team Dragons', min_value=0, max_value=1, value=0)
red_dragons = st.sidebar.number_input('Red Team Dragons', min_value=0, max_value=1, value=0)
herald = st.sidebar.checkbox('Blue Team secured Rift Herald')
first_blood = st.sidebar.checkbox('Blue Team secured First Blood')
herald_int = int(herald)
first_blood_int = int(first_blood)

if herald_int == 0:
    objective_diff = blue_dragons - red_dragons - 1
else:
    objective_diff = blue_dragons - red_dragons + 1

input_data = pd.DataFrame({
    'gold_diff': [gold_diff],
    'xp_diff': [xp_diff],
    'cs_diff': [cs_diff],
    'objective_diff': [objective_diff],
    'first_blood': [first_blood_int],
    'herald_diff': [herald_int] 
})

if st.button('Predict Match Outcome'):
    # calculate win prob
    win_prob = model.predict_proba(input_data)[0][1] * 100
    loss_prob = 100 - win_prob

    st.subheader('Output')
    col1, col2 = st.columns(2)

    col1.metric('Blue Team Win Probability', f'{win_prob:.1f}%')

    if win_prob > 60:
        st.success('Blue Team is very ahead and likely to win the game')
    elif win_prob < 40:
        st.error('Blue Team is far behind and is likely to loss the game')
    else:
        st.info('The team is tightly contested and volatile')