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
# DARK THEME CSS
# ------------------------------------------------

st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #0f172a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Text */
h1, h2, h3, h4, h5, h6,
p, li, label, span {
    color: white !important;
}

/* Cards */
.custom-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

/* Prediction Card */
.prediction-card {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-size: 18px;
    font-weight: bold;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("stack_classifier.pkl")

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("🌸 Stacking Classification on Iris Dataset")

st.markdown("""
<div class="custom-card">

<h3>🤖 Ensemble Learning Model</h3>

<b>Base Learners</b>

<ul>
<li>🌲 Random Forest Classifier</li>
<li>📈 Support Vector Machine (SVM)</li>
<li>🚀 Gradient Boosting Classifier</li>
</ul>

<b>Meta Learner</b>

<ul>
<li>🧠 Logistic Regression</li>
</ul>

Predictions from multiple models are combined and passed to a meta learner
to improve classification performance.

</div>
""", unsafe_allow_html=True)

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

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown("""
### Target Classes

- Setosa
- Versicolor
- Virginica
""")

# ------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------

st.sidebar.header("🌿 Flower Measurements")

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

st.subheader("📋 Input Values")

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
        0: "🌸 Setosa",
        1: "🌿 Versicolor",
        2: "🌺 Virginica"
    }

    flower = flower_names[prediction]

    st.markdown(
        f"""
        <div class="prediction-card">
            Predicted Flower Species<br><br>
            {flower}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Flower": [
            "Setosa",
            "Versicolor",
            "Virginica"
        ],
        "Probability (%)":
        np.round(probability[0] * 100, 2)
    })

    st.dataframe(
        prob_df,
        use_container_width=True
    )

    st.bar_chart(
        prob_df.set_index("Flower")
    )

# ------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------

st.markdown("---")

st.subheader("📚 About Stacking Ensemble")

st.markdown("""
Stacking is an ensemble learning technique where multiple machine learning models are trained independently.

Their predictions are then used as inputs to a final model called the meta learner.

This often achieves better performance than using a single model.
""")

st.info("""
Hyperparameter Tuning Used

• Random Forest → GridSearchCV

• SVM → GridSearchCV

Final Ensemble

Random Forest + SVM + Gradient Boosting

↓

Logistic Regression

↓

Final Prediction
""")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🌸 Machine Learning Project | Stacking Classification on Iris Dataset
    </div>
    """,
    unsafe_allow_html=True
)