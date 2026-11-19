# Student Pass Prediction and Algorithm Comparison

A complete, beginner-friendly machine learning project in Python featuring both an interactive **Streamlit Web Application** and a **Command-Line Interface**, comparing three supervised classification algorithms:
1. **Decision Tree (using Gini Index)**
2. **Random Forest**
3. **Support Vector Machine (SVM with Feature Scaling via Pipeline)**

---

## 📌 Project Overview

This project predicts whether a student will **Pass (1)** or **Fail (0)** based on three academic indicators:
- **Weekly Study Hours** (hours spent studying per week)
- **Attendance Percentage** (percentage of classes attended, 0–100%)
- **Previous Exam Score** (prior exam mark, 0–100)

All three models are trained and tested on the exact same train/test split to guarantee a fair, apples-to-apples comparison.

> ⚠️ **Important Educational Disclaimer:**
> The included `student_data.csv` is a small synthetic sample dataset created specifically for coursework and educational demonstration. Small sample datasets cannot be used to make reliable real-world academic predictions or decisions. It is designed to illustrate machine learning workflows, pipelines, and evaluation metrics.

---

## 🗂️ Project File Structure

```text
ml project/
│
├── student_data.csv    # Labeled dataset with 60 student records
├── requirements.txt    # Python dependencies (pandas, scikit-learn, streamlit)
├── app.py              # NEW: Streamlit interactive web interface
├── main.py             # Original CLI script with training & evaluation
└── README.md           # Documentation, explanations, and run instructions
```

---

## 🧠 Algorithms Explained (For Your Class Presentation)

### 1. Decision Tree (Gini Index)
- **Concept:** A flowchart-like tree structure where nodes test an attribute, branches represent decision outcomes, and leaves represent final class labels (`Pass` or `Fail`).
- **Gini Index:** Measures impurity or disorder. A Gini score of `0` denotes a completely pure node (all students in that branch share the same outcome). The algorithm chooses splits that minimize the Gini index:
  $$\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$$
- **Pros:** Highly interpretable and intuitive to explain.
- **Cons:** Can overfit if not regularized.

### 2. Random Forest
- **Concept:** An ensemble method that constructs a "forest" of multiple independent decision trees (100 trees in this project) trained on random data subsets.
- **Voting Mechanism:** Each tree makes an individual prediction; the forest outputs the class that receives the majority vote.
- **Pros:** Significantly reduces overfitting compared to a single decision tree.

### 3. Support Vector Machine (SVM)
- **Concept:** A geometric classifier that finds the optimal decision boundary (hyperplane) maximizing the margin between the closest data points of opposing classes (support vectors).
- **Why Feature Scaling via Pipeline is Essential:**
  - SVM calculates Euclidean distances between points.
  - Features with wide ranges (e.g., Previous Exam Score `0–100`) would dominate features with smaller ranges (e.g., Study Hours `1–16`) if left unscaled.
  - A scikit-learn `Pipeline` with `StandardScaler` standardizes features to a mean of $0$ and standard deviation of $1$. Crucially, the pipeline learns scaling statistics only from the training set, preventing **data leakage** into the test set or new user inputs.

---

## ⚙️ Installation & Setup (VS Code Terminal)

Follow these steps in your VS Code terminal:

### Step 1: Open the Terminal in VS Code
- In VS Code, press `Ctrl + ~` (or select menu **Terminal > New Terminal**).
- Ensure your terminal path is inside your project folder:
  ```powershell
  c:\Users\REHAN\OneDrive\Desktop\ml project
  ```

### Step 2: Install Required Libraries
Run:
```powershell
pip install -r requirements.txt
```
*(This installs `pandas`, `scikit-learn`, and `streamlit`.)*

---

## 🌐 How to Run the Web Application (Streamlit)

Start the web application by running:
```powershell
streamlit run app.py
```

### What Happens:
1. Streamlit compiles and launches the web app.
2. Your default web browser will automatically open to `http://localhost:8501`.
3. If it does not open automatically, hold `Ctrl` and click the link displayed in the terminal.

### Web App Features:
- **Live Student Prediction:** Enter Study Hours, Attendance %, and Previous Score; click **Predict** to view real-time Pass/Fail badges from all three algorithms.
- **Input Validation:** Clear warnings and error messages prevent negative values or out-of-range percentages.
- **Accuracy Comparison Table & Chart:** A side-by-side summary table and an interactive bar chart comparing the accuracy of all three algorithms.
- **Confusion Matrices:** Interactive tabs for each algorithm displaying True Negatives, False Positives, False Negatives, and True Positives with beginner-friendly explanations.
- **Raw Data Viewer:** An expandable section allowing you and your instructor to inspect the dataset directly in the browser.

To stop the web server, press `Ctrl + C` in the VS Code terminal.

---

## 💻 Running the Command-Line Version (Optional)

If you ever need to demonstrate the original CLI version, simply run:
```powershell
python main.py
```

---

## 💡 Key Machine Learning Questions You May Be Asked in Class

1. **Why split data into Train and Test sets?**  
   To evaluate how well models generalize to unseen data, preventing false confidence from models simply memorizing training records.
2. **Why evaluate all models on the identical split?**  
   Using the exact same split (`random_state=42`) provides a controlled, fair benchmark across all algorithms.
3. **What is a Confusion Matrix?**  
   - **TN (True Negative):** Student who failed and was correctly predicted as Fail.
   - **FP (False Positive):** Student who failed but was predicted as Pass (Type I error).
   - **FN (False Negative):** Student who passed but was predicted as Fail (Type II error).
   - **TP (True Positive):** Student who passed and was correctly predicted as Pass.
