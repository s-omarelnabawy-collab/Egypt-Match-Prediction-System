import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_models():
    # Load Logistic Model (Outcome)
    with open('log_model.pkl', 'rb') as f:
        log_data = pickle.load(f)

    with open('lin_model.pkl', 'rb') as f:
        lin_data = pickle.load(f)

    return log_data, lin_data

try:
    log_data, lin_data = load_models()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found! Please run 'linear_reg.py' and 'log_reg.py' first so they can generate the .pkl files.")
    st.stop()

st.set_page_config(page_title="Egypt National Team Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Egypt Match Prediction System")

st.header("📋 General Match Features (Shared)")
col1, col2 = st.columns(2)

with col1:
    home_input = st.selectbox("Match Venue", options=["Home (In Egypt)", "Away / Neutral"])
    home = 1 if "Home" in home_input else 0

    fifa_rank = st.number_input("Opponent FIFA Rank", min_value=1, max_value=211, value=50)
    opp_value = st.number_input("Opponent Market Value (M€)", min_value=0.0, value=45.0, step=0.5)

with col2:
    egy_form = st.slider("Egypt Current Form", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
    salah_input = st.radio("Is Mohamed Salah playing?", options=["Yes", "No"])
    salah = 1 if salah_input == "Yes" else 0

st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Goals Predictor (Linear)", "🏆 Outcome Predictor (Logistic)"])

with tab1:
    st.subheader("📊 Features for Goals Model")
    col1_reg, col2_reg = st.columns(2)
    with col1_reg:
        rest_days = st.number_input("Egypt Rest Days", min_value=1, max_value=15, value=4)
    with col2_reg:
        opp_xga = st.number_input("Opponent xGA (Expected Goals Against)", min_value=0.0, max_value=4.0, value=1.5,
                                  step=0.1)

    if st.button("Predict Goals ⚽"):
        # [Home, FIFA_Rank, Egy_Form, Rest_Days, Opp_xGA, Opp_Value, Salah]
        features_lin = np.array([[home, fifa_rank, egy_form, rest_days, opp_xga, opp_value, salah]])

        # Transform and Predict
        data_scaled = lin_data['scaler'].transform(features_lin)
        data_poly = lin_data['poly'].transform(data_scaled)
        pred_goals = lin_data['model'].predict(data_poly)[0]

        final_goals = max(0, int(round(pred_goals)))

        st.success(
            f"🎯 **Predicted Goals for Egypt: {final_goals}** (Exact mathematical calculation: {pred_goals:.2f} goals)")

with tab2:
    st.subheader("📊 Features for Outcome Model")

    opp_form = st.slider("Opponent Current Form", min_value=0.0, max_value=3.0, value=1.4, step=0.1)
    egy_xg = st.slider("Egypt xG (Expected Goals)", min_value=0.0, max_value=4.0, value=1.85, step=0.05)
    egy_xga_clf = st.slider("Egypt xGA (Expected Goals Against)", min_value=0.0, max_value=4.0, value=0.95, step=0.05)

    if st.button("Predict Outcome 🏆"):
        # [Home, FIFA_Rank, Opp_Value, Egy_Form, Opp_Form, Egy_xG, Egy_xGA, Salah]
        features_log = np.array([[home, fifa_rank, opp_value, egy_form, opp_form, egy_xg, egy_xga_clf, salah]])

        data_scaled = log_data['scaler'].transform(features_log)
        data_poly = log_data['poly'].transform(data_scaled)
        pred_outcome = log_data['model'].predict(data_poly)[0]

        if pred_outcome == 1:
            st.success("🏆 **Match outcome prediction: Egypt Wins!** (Predicted Class: 1)")
            st.balloons()
        else:
            st.warning("⚠️ **Match outcome prediction: Loss or Draw for Egypt.** (Predicted Class: 0)")