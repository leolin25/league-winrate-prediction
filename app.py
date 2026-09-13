import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("League of Legends: 10-Minute Win Predictor")

st.sidebar.header("Match Context")
st.warning("**Disclaimer:** This model is trained on Season 10 data from 2020. The current meta, map objectives, and champion baselines have changed significantly.")
match_date = st.sidebar.date_input("Match Date")

st.sidebar.header("10-Minute Differentials")

# add 10 min differential field for user to fill

st.subheader(f"Blue Team Win Probability: {blue_win_prob * 100:.1f}%")

# Chart Layout
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(["Blue Team", "Red Team"], [blue_win_prob, red_win_prob], color=['#3498db', '#e74c3c'])
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title(f"Win Probability Forecast (Match: {match_date})")
ax.bar_label(bars, fmt='%.2f', padding=3)

st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

st.download_button(
    label="Save & Download Chart",
    data=buf,
    file_name=f"win_probability_{match_date}.png",
    mime="image/png"
)