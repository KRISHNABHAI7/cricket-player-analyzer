import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Cricket Player Performance Analyzer",
    page_icon="🏏",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏏 Cricket Player Performance Analyzer")

st.write(
    """
    This Machine Learning project predicts:

    • Future Batting Average using Linear Regression  
    • Elite Player Probability using Logistic Regression
    """
)

st.markdown("---")

# =========================================================
# CREATE SYNTHETIC DATASET
# =========================================================

np.random.seed(42)

n = 300

batting_avg = np.random.uniform(15, 60, n)

consistency_idx = np.random.uniform(10, 40, n)

form_trend = np.random.uniform(-10, 10, n)

strike_rate = np.random.uniform(50, 120, n)

experience = np.random.uniform(2, 6, n)

clutch_score = (
    batting_avg * 0.4
    + strike_rate * 0.3
    + consistency_idx * 0.3
)

boundary_eff = np.random.uniform(20, 80, n)

future_avg = (
    batting_avg * 0.6
    + consistency_idx * 0.2
    + form_trend * 0.2
)

is_elite = (
    (batting_avg > 40)
    & (consistency_idx > 25)
    & (clutch_score > 45)
).astype(int)

df = pd.DataFrame({

    'batting_avg': batting_avg,
    'consistency_idx': consistency_idx,
    'form_trend': form_trend,
    'strike_rate': strike_rate,
    'experience': experience,
    'clutch_score': clutch_score,
    'boundary_eff': boundary_eff,
    'future_avg': future_avg,
    'is_elite': is_elite

})

# =========================================================
# FEATURES
# =========================================================

features = [

    'batting_avg',
    'consistency_idx',
    'form_trend',
    'strike_rate',
    'experience',
    'clutch_score',
    'boundary_eff'

]

# =========================================================
# LINEAR REGRESSION MODEL
# =========================================================

X = df[features]

y = df['future_avg']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr_scaler = StandardScaler()

X_train_scaled = lr_scaler.fit_transform(X_train)

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)

# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================

X = df[features]

y = df['is_elite']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

log_scaler = StandardScaler()

X_train_scaled = log_scaler.fit_transform(X_train)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

# =========================================================
# USER INPUT SECTION
# =========================================================

st.header("📊 Enter Player Statistics")

# -------------------------

batting_avg_input = st.number_input(
    "Batting Average",
    min_value=0.0,
    max_value=100.0,
    value=45.0
)

st.caption("Average runs scored per dismissal.")

# -------------------------

consistency_input = st.number_input(
    "Consistency Index",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

st.caption("Measures how consistently the player performs.")

# -------------------------

form_input = st.number_input(
    "Form Trend",
    min_value=-20.0,
    max_value=20.0,
    value=5.0
)

st.caption("Shows whether current form is improving or declining.")

# -------------------------

strike_input = st.number_input(
    "Strike Rate",
    min_value=0.0,
    max_value=200.0,
    value=85.0
)

st.caption("Runs scored per 100 balls.")

# -------------------------

experience_input = st.number_input(
    "Experience",
    min_value=0.0,
    max_value=20.0,
    value=4.0
)

st.caption("Represents overall experience level.")

# -------------------------

clutch_input = st.number_input(
    "Clutch Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

st.caption("Performance in pressure situations.")

# -------------------------

boundary_input = st.number_input(
    "Boundary Efficiency",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

st.caption("Efficiency in hitting boundaries.")

st.markdown("---")

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Performance"):

    new_player = {

        'batting_avg': batting_avg_input,
        'consistency_idx': consistency_input,
        'form_trend': form_input,
        'strike_rate': strike_input,
        'experience': experience_input,
        'clutch_score': clutch_input,
        'boundary_eff': boundary_input

    }

    input_df = pd.DataFrame(
        [[new_player[f] for f in features]],
        columns=features
    )

    # =====================================================
    # LINEAR REGRESSION PREDICTION
    # =====================================================

    scaled_lr = lr_scaler.transform(input_df)

    future_prediction = lr_model.predict(scaled_lr)[0]

    # =====================================================
    # LOGISTIC REGRESSION PREDICTION
    # =====================================================

    scaled_log = log_scaler.transform(input_df)

    elite_probability = log_model.predict_proba(scaled_log)[0][1]

    # =====================================================
    # RESULTS
    # =====================================================

    st.success("Prediction Completed Successfully")

    st.header("🏏 Scouting Report")

    st.write(f"### Predicted Future Average : {future_prediction:.2f}")

    st.write(f"### Elite Probability : {elite_probability:.2%}")

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if elite_probability > 0.5:

        st.success("🌟 ELITE PLAYER")

    else:

        st.warning("📊 DEVELOPING PLAYER")

    st.markdown("---")

    st.subheader("📌 Model Information")

    st.write("• Linear Regression predicts future batting average.")

    st.write("• Logistic Regression predicts elite player probability.")

    st.write("• Dataset used is synthetic cricket performance data.")