import streamlit as st
import joblib
from catboost import CatBoostRegressor

# -------------------------------
# Load models and vectorizer
# -------------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
rf_clf = joblib.load("rf_classifier.pkl")

cat_reg = CatBoostRegressor()
cat_reg.load_model("catboost_regressor.cbm")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AutoJudge (RF)", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Programming Problem Difficulty Predictor (Random Forest)")

st.write(
    "Enter the programming problem details below. "
    "The system predicts difficulty based on textual complexity."
)

problem_desc = st.text_area(
    "Problem Description",
    height=150
)

input_desc = st.text_area(
    "Input Description",
    height=120
)

output_desc = st.text_area(
    "Output Description",
    height=120
)

if st.button("Predict Difficulty"):
    if not problem_desc.strip() or not input_desc.strip() or not output_desc.strip():
        st.warning("Please fill in all fields.")
    else:
        combined_text = (
            problem_desc + " " +
            input_desc + " " +
            output_desc
        )

        X_input = tfidf.transform([combined_text])

        # Classification (Random Forest)
        class_pred = rf_clf.predict(X_input)[0]

        # Regression (CatBoost)
        score_pred = cat_reg.predict(X_input)[0]

        st.success(f"📌 Predicted Difficulty Class: **{class_pred.upper()}**")
        st.info(f"📊 Predicted Difficulty Score: **{score_pred:.2f}**")

        st.caption(
            "Model used: Random Forest (classification) + CatBoost (regression)"
        )
