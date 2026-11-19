"""
=============================================================================
Project: Student Pass Prediction & Algorithm Comparison
Course Project: Beginner Machine Learning in Python
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

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


def load_dataset(file_path="student_data.csv"):
    """
    Step 1: Load the dataset from CSV file.
    Pandas reads the tabular data into a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] Successfully loaded dataset: '{file_path}'")
        print(f"[INFO] Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns.\n")
        return data
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{file_path}'. Please make sure it is in the same folder.")
        sys.exit(1)


def prepare_data(data):
    """
    Step 2: Separate features (X) and target (y), then split into Train/Test sets.
    - X: The input features used for learning (study_hours, attendance_percentage, previous_exam_score).
    - y: The target outcome to predict (passed: 0 for Fail, 1 for Pass).
    - test_size=0.2 means 80% data is for training and 20% for testing.
    - random_state=42 guarantees reproducibility (everyone gets the same split).
    """
    feature_cols = ["study_hours", "attendance_percentage", "previous_exam_score"]
    target_col = "passed"

    X = data[feature_cols]
    y = data[target_col]

    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Testing set size : {len(X_test)} samples\n")
    return X_train, X_test, y_train, y_test, feature_cols


def build_models():
    """
    Step 3: Initialize the three machine learning models.
    
    1. Decision Tree (criterion='gini'):
       Uses the Gini Impurity index to decide how to branch and split data.
       
    2. Random Forest:
       An ensemble of multiple decision trees that vote on the final prediction,
       reducing overfitting and improving stability.
       
    3. Support Vector Machine (SVM):
       Finds the optimal hyperplane that separates Pass and Fail classes.
       *CRITICAL*: SVM is sensitive to feature scale (e.g., hours [1-16] vs score [30-95]).
       We use a scikit-learn Pipeline with StandardScaler to ensure features
       are properly scaled without data leakage.
    """
    models = {
        "Decision Tree (Gini)": DecisionTreeClassifier(
            criterion="gini", 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        ),
        "Support Vector Machine (SVM)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=42))
        ])
    }
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Step 4: Train each model on the SAME training data and evaluate on the SAME test data.
    This ensures an apples-to-apples, fair comparison.
    """
    results = {}

    print("=" * 70)
    print(" MODEL EVALUATION & CONFUSION MATRICES ")
    print("=" * 70)

    for name, model in models.items():
        # Train (fit) the model
        model.fit(X_train, y_train)

        # Make predictions on the unseen test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm
        }

        # Display confusion matrix clearly
        # Format:
        #                 Predicted Fail (0)   Predicted Pass (1)
        # Actual Fail (0)   [TN]                 [FP]
        # Actual Pass (1)   [FN]                 [TP]
        tn, fp, fn, tp = cm.ravel()
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix:")
        print(f"  True Negatives  (Actual Fail, Predicted Fail): {tn}")
        print(f"  False Positives (Actual Fail, Predicted Pass): {fp}")
        print(f"  False Negatives (Actual Pass, Predicted Fail): {fn}")
        print(f"  True Positives  (Actual Pass, Predicted Pass): {tp}")

    return results


def print_comparison_table(results):
    """
    Step 5: Print a clean comparison table summarizing the accuracy of all 3 models.
    """
    table_data = []
    for name, res in results.items():
        table_data.append({
            "Algorithm": name,
            "Accuracy (%)": f"{res['accuracy'] * 100:.2f}%",
            "Correct Predictions": f"{res['confusion_matrix'].diagonal().sum()} / {res['confusion_matrix'].sum()}"
        })

    df_results = pd.DataFrame(table_data)
    print("\n" + "=" * 70)
    print(" ALGORITHM COMPARISON SUMMARY ")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70 + "\n")


def interactive_prediction(results, feature_cols):
    """
    Step 6: Allow the user to input student values and observe predictions from all 3 models.
    """
    print("-" * 70)
    print(" LIVE PREDICTION TOOL ")
    print("Enter student details to see predictions from all 3 models.")
    print("Type 'exit' or 'q' at any prompt to finish.")
    print("-" * 70)

    while True:
        try:
            print("\n--- Enter New Student Details ---")
            raw_hours = input("1. Weekly Study Hours (e.g., 2 - 20): ").strip()
            if raw_hours.lower() in ["exit", "q"]:
                break

            raw_attendance = input("2. Attendance Percentage (0 - 100%): ").strip()
            if raw_attendance.lower() in ["exit", "q"]:
                break

            raw_score = input("3. Previous Exam Score (0 - 100): ").strip()
            if raw_score.lower() in ["exit", "q"]:
                break

            # Convert to float and validate ranges
            hours = float(raw_hours)
            attendance = float(raw_attendance)
            score = float(raw_score)

            if hours < 0 or attendance < 0 or attendance > 100 or score < 0 or score > 100:
                print("[WARNING] Please enter realistic values (hours >= 0, percentage 0-100, score 0-100).")
                continue

            # Create a 1-row DataFrame matching the training feature names
            student_df = pd.DataFrame([[hours, attendance, score]], columns=feature_cols)

            print("\n" + "." * 50)
            print(f"Inputs: Study Hours={hours}, Attendance={attendance}%, Previous Score={score}")
            print("Model Predictions:")
            print("." * 50)

            for name, res in results.items():
                pred = res["model"].predict(student_df)[0]
                status = "PASS" if pred == 1 else "FAIL"
                print(f"  * {name:<30}: {status}")
            print("." * 50)

        except ValueError:
            print("[ERROR] Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting prediction mode.")
            break

    print("\nThank you for using the Student Pass Prediction system!")


def main():
    print("""
#############################################################################
#             STUDENT PASS PREDICTION & ALGORITHM COMPARISON                #
#      Decision Tree (Gini) vs. Random Forest vs. Scaled SVM (Pipeline)     #
#############################################################################
    """)
    print("[NOTE] This project uses a synthetic sample dataset designed for")
    print("       educational purposes. Real-world academic forecasting requires")
    print("       larger, verified datasets covering multiple terms and factors.\n")

    # Step 1: Load data
    data = load_dataset("student_data.csv")

    # Step 2: Prepare & Split
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(data)

    # Step 3: Build models
    models = build_models()

    # Step 4: Train & Evaluate
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # Step 5: Summary Table
    print_comparison_table(results)

    # Step 6: Live Prediction Loop
    interactive_prediction(results, feature_cols)


if __name__ == "__main__":
    main()
