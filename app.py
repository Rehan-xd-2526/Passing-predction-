"""
=============================================================================
Project: Student Pass Prediction & Algorithm Comparison (Web App)
Built with: Streamlit, pandas, and scikit-learn
-----------------------------------------------------------------------------
Features:
  1. Study Hours per week
  2. Attendance Percentage (0 - 100%)
  3. Previous Exam Score (0 - 100)
Target:
  Pass (1) or Fail (0)

Algorithms Compared:
  1. Decision Tree (using Gini Index)
  2. Random Forest
  3. Support Vector Machine (SVM with StandardScaler via Pipeline)
=============================================================================
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# 1. Page Configuration & Header
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Pass Prediction & Model Comparison",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Pass Prediction & Algorithm Comparison")
st.markdown(
    """
    This beginner-friendly machine learning web application compares **three classification algorithms** 
    to predict whether a student is likely to **Pass** or **Fail** based on academic habits.
    """
)

# Prominent Educational Disclaimer
st.warning(
    "⚠️ **Educational Disclaimer:** This application uses a small synthetic sample dataset created "
    "specifically for coursework and classroom demonstrations. The predictions are **not reliable** "
    "for real-world student decisions or academic evaluations."
)

# ---------------------------------------------------------------------------
# 2. Machine Learning Pipeline (Cached for fast performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_and_train_models(file_path="student_data.csv"):
    """
    Loads dataset, performs train-test split, and trains all 3 models on the SAME split.
    Using @st.cache_resource ensures models train only once when the app starts,
    rather than retraining on every button click.
    """
    # Step A: Load Dataset
    data = pd.read_csv(file_path)
    feature_cols = ["study_hours", "attendance_percentage", "previous_exam_score"]
    target_col = "passed"

    X = data[feature_cols]
    y = data[target_col]

    # Step B: Train/Test Split (80% train, 20% test, fixed random_state for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step C: Define the 3 Models
    # 1. Decision Tree using Gini Impurity index
    # 2. Random Forest ensemble of 100 trees
    # 3. SVM with StandardScaler inside a scikit-learn Pipeline (prevents data leakage)
    models = {
        "Decision Tree (Gini)": DecisionTreeClassifier(criterion="gini", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=42))
        ])
    }

    # Step D: Train & Evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm
        }

    return data, results, feature_cols, len(X_train), len(X_test)

# Load data and train models
try:
    dataset, model_results, feature_columns, train_count, test_count = load_and_train_models()
except FileNotFoundError:
    st.error("❌ Could not find `student_data.csv`. Please make sure it exists in the project folder.")
    st.stop()

# ---------------------------------------------------------------------------
# 3. Interactive Prediction Section
# ---------------------------------------------------------------------------
st.header("1. Live Student Prediction")
st.markdown("Enter student details below and click **Predict** to view each algorithm's verdict.")

# Create input form with 3 columns for side-by-side inputs
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.number_input(
            "Weekly Study Hours",
            min_value=0.0,
            max_value=80.0,
            value=8.0,
            step=0.5,
            help="Number of hours dedicated to studying per week (realistic range: 0 - 80)."
        )

    with col2:
        attendance = st.number_input(
            "Attendance Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
            help="Percentage of attended classes (0% to 100%)."
        )

    with col3:
        previous_score = st.number_input(
            "Previous Exam Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Score obtained in the previous exam (0 to 100)."
        )

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# Process prediction and input validation
if submitted:
    # Input Validation
    validation_passed = True

    if study_hours < 0:
        st.error("Study hours cannot be negative.")
        validation_passed = False
    elif study_hours > 70:
        st.warning("⚠️ Note: Study hours over 70 hours/week is unusually high, but prediction will still proceed.")

    if not (0 <= attendance <= 100):
        st.error("Attendance percentage must be between 0 and 100.")
        validation_passed = False

    if not (0 <= previous_score <= 100):
        st.error("Previous exam score must be between 0 and 100.")
        validation_passed = False

    if validation_passed:
        # Create input DataFrame matching training feature names
        input_data = pd.DataFrame(
            [[study_hours, attendance, previous_score]],
            columns=feature_columns
        )

        st.subheader("Prediction Results:")
        pred_cols = st.columns(3)

        for i, (name, res) in enumerate(model_results.items()):
            pred = res["model"].predict(input_data)[0]
            with pred_cols[i]:
                st.markdown(f"### {name}")
                if pred == 1:
                    st.success("🎉 **PASS**")
                else:
                    st.error("⚠️ **FAIL**")

        st.caption(
            f"Input values evaluated: Study Hours = **{study_hours} hrs/wk**, "
            f"Attendance = **{attendance}%**, Previous Score = **{previous_score}**"
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# 4. Model Comparison: Accuracy & Chart
# ---------------------------------------------------------------------------
st.header("2. Model Accuracy Comparison")
st.markdown(
    f"All models were evaluated on the **same test split** "
    f"({test_count} unseen test students from a total of {train_count + test_count} records)."
)

# Build summary comparison table
comparison_rows = []
for name, res in model_results.items():
    cm = res["confusion_matrix"]
    correct = cm.diagonal().sum()
    total = cm.sum()
    comparison_rows.append({
        "Algorithm": name,
        "Accuracy (%)": round(res["accuracy"] * 100, 2),
        "Correct Test Predictions": f"{correct} / {total}"
    })

comparison_df = pd.DataFrame(comparison_rows)

col_table, col_chart = st.columns([1, 1])

with col_table:
    st.subheader("Summary Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

with col_chart:
    st.subheader("Accuracy Comparison Chart")
    chart_data = pd.DataFrame({
        "Algorithm": [row["Algorithm"] for row in comparison_rows],
        "Accuracy (%)": [row["Accuracy (%)"] for row in comparison_rows]
    }).set_index("Algorithm")
    st.bar_chart(chart_data)

st.markdown("---")

# ---------------------------------------------------------------------------
# 5. Confusion Matrices Section
# ---------------------------------------------------------------------------
st.header("3. Confusion Matrices")
st.markdown(
    """
    A **Confusion Matrix** shows how many test samples each model classified correctly or incorrectly:
    - **True Negative (TN):** Actually Failed & Predicted Failed *(Correct)*
    - **False Positive (FP):** Actually Failed & Predicted Passed *(Type I error)*
    - **False Negative (FN):** Actually Passed & Predicted Failed *(Type II error)*
    - **True Positive (TP):** Actually Passed & Predicted Passed *(Correct)*
    """
)

cm_tabs = st.tabs(list(model_results.keys()))

for i, (name, res) in enumerate(model_results.items()):
    with cm_tabs[i]:
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        cm_table = pd.DataFrame(
            [
                [f"TN = {tn}", f"FP = {fp}"],
                [f"FN = {fn}", f"TP = {tp}"]
            ],
            index=["Actual: Fail (0)", "Actual: Pass (1)"],
            columns=["Predicted: Fail (0)", "Predicted: Pass (1)"]
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.table(cm_table)
        with c2:
            st.write(f"- **Accuracy:** {res['accuracy'] * 100:.2f}%")
            st.write(f"- **Correctly Identified Failures (TN):** {tn}")
            st.write(f"- **Correctly Identified Passes (TP):** {tp}")
            st.write(f"- **False Alarms (FP):** {fp}")
            st.write(f"- **Missed Passes (FN):** {fn}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 6. Dataset Viewer (Optional inspection for students)
# ---------------------------------------------------------------------------
with st.expander("🔍 View Raw Training Dataset (student_data.csv)"):
    st.write(f"Total rows: {dataset.shape[0]} | Columns: {dataset.shape[1]}")
    st.dataframe(dataset, use_container_width=True)
