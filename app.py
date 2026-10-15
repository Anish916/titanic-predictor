import joblib
import pandas as pd
import streamlit as st

from train_model import FEATURES, PORT_MAP, SEX_MAP

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.caption(
    "Enter passenger details and the model estimates the chance of survival. "
    "XGBoost trained on the Kaggle Titanic dataset, 83% accuracy on held-out data."
)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    age = st.slider("Age", 0, 80, 30)
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 6, 0)
    fare = st.number_input("Fare paid (£)", 0.0, 520.0, 15.0, step=1.0)
    port = st.selectbox(
        "Port of embarkation",
        {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}.items(),
        format_func=lambda kv: kv[1],
        index=2,
    )[0]

if st.button("Predict survival", type="primary"):
    row = pd.DataFrame(
        [[pclass, SEX_MAP[sex], age, sibsp, parch, fare, PORT_MAP[port]]],
        columns=FEATURES,
    )
    prob = float(model.predict_proba(row)[0][1])
    if prob >= 0.5:
        st.success(f"Likely to survive ({prob:.0%} probability)")
    else:
        st.error(f"Unlikely to survive ({prob:.0%} probability)")
    st.progress(prob)
    st.caption(
        "Historical model, not a judgement about any real person. "
        "Trained on 891 passengers, so unusual combinations are estimated from very few examples."
    )
