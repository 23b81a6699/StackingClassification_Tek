import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Stacking Classification",
    page_icon="🌸",
    layout="wide"
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("stack_classifier.pkl")

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("🌸 Stacking Classification on Iris Dataset")

st.markdown("""
This project demonstrates **Stacking Ensemble Learning**.

### Base Learners
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier

### Meta Learner
- Logistic Regression

The predictions from the base learners are combined and passed to the meta learner,
which makes the final prediction.
""")

# ------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------

with st.expander("📊 Iris Dataset Information"):

    iris = load_iris()

    st.write("Dataset Shape:", iris.data.shape)

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    st.dataframe(df.head())

    st.write("""
    **Target Classes**
    - Setosa
    - Versicolor
    - Virginica
    """)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    5.0,
    3.5
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.4
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    3.0,
    0.2
)

# ------------------------------------------------
# DISPLAY INPUTS
# ------------------------------------------------

st.subheader("Input Values")

input_df = pd.DataFrame({
    "Feature": [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ],
    "Value": [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
})

st.table(input_df)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("🔍 Predict Flower"):

    data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)

    flower_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    flower = flower_names[prediction]

    st.success(
        f"Predicted Flower Species: {flower}"
    )

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Flower": [
            "Setosa",
            "Versicolor",
            "Virginica"
        ],
        "Probability (%)":
        probability[0] * 100
    })

    st.dataframe(prob_df)

# ------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------

st.markdown("---")

st.subheader("📚 About Stacking")

st.write("""
Stacking is an ensemble learning technique where multiple machine learning models
(base learners) are trained first.

The predictions from these models are then used as inputs to another model
called the meta learner.

This often improves prediction accuracy compared to individual models.
""")

st.info("""
Hyperparameter Tuning Used:
- Random Forest: GridSearchCV
- SVM: GridSearchCV

Final Ensemble:
Random Forest + SVM + Gradient Boosting → Logistic Regression
""")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("Machine Learning Project - Stacking Classification on Iris Dataset")