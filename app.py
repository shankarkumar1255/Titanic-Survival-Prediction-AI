import streamlit as st
import joblib
import pandas as pd
from llm_feature import call_llm

# Load model
model = joblib.load("models/titanic_model.pkl")

st.sidebar.title("AI/ML Capstone Project")
st.sidebar.write("Student: Shankar Kumar")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.title("🚢 Titanic Survival Prediction")
st.metric("Model Accuracy", "85.47%")
st.caption("Dataset: Titanic (891 Records)")
st.divider()
st.markdown("""
### Machine Learning Based Titanic Survival Prediction
Fill the passenger details below and click **Predict** to know the survival prediction.
""")

passenger_id = st.number_input("Passenger ID", min_value=1, value=892)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
sibsp = st.number_input("Siblings/Spouse", min_value=0, value=0)
parch = st.number_input("Parents/Children", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=7.25)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
age_group = st.selectbox("Age Group", ["Child", "Teen", "Adult", "Senior"])

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

sex = 0 if sex == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]
age_group = {"Child": 0, "Teen": 1, "Adult": 2, "Senior": 3}[age_group]
title = 0

input_data = pd.DataFrame([[passenger_id, pclass, sex, sibsp, parch, fare,
                            embarked, age_group, family_size, is_alone, title]])
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
        st.info(f"Survival Probability: {probability[0][1]*100:.2f}%")
        st.progress(float(probability[0][1]))
    else:
        st.error("❌ Passenger Did Not Survive")
        st.info(f"Survival Probability: {probability[0][1]*100:.2f}%")
        st.progress(float(probability[0][1]))



    st.divider()
    st.subheader("🤖 AI Explanation")

    prompt = f"""
    Passenger Details:
    - Class: {pclass}
    - Sex: {"Male" if sex == 0 else "Female"}
    - Fare: {fare}
    - Family Size: {family_size}
    - Is Alone: {is_alone}

    Prediction:
    {"Survived" if prediction[0] == 1 else "Did Not Survive"}

    Explain this prediction in simple English in 3-4 lines.
    """

    explanation = call_llm(
        "You are a helpful AI assistant for Machine Learning.",
        prompt
    )

    st.write(explanation)