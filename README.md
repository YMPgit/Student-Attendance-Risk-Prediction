# Student Attendance Risk Prediction

> **IBM SkillsBuild Internship Project**
> Author: Yash Patade | Algorithm: Decision Tree Classifier

A beginner-friendly, end-to-end machine learning project that predicts whether a student will **attend** (`1`) or be **absent** (`0`) from class, built with a Decision Tree classifier on a real Kaggle dataset. Clean, reproducible, and suitable for an IBM internship submission.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Student Attendance Prediction Dataset — College Level |
| **Source** | [Kaggle — kundanbedmutha/student-attendance-dataset-college-level](https://www.kaggle.com/datasets/kundanbedmutha/student-attendance-dataset-college-level) |
| **File** | `Attendance_Prediction.csv` |
| **Records** | 20,000 rows × 15 columns |
| **Target** | `attendance` — binary (1 = Present, 0 = Absent) |

---

## Project Structure

```
.
├── YashPatade_student_attendance_risk_prediction.ipynb    # Main Jupyter notebook (EDA → Training → Results)
├── Attendance_Prediction.csv                              # Real dataset from Kaggle
├── requirements.txt                                       # Python dependencies
├── projectReport.docx                                     # Full internship project report
└── README.md                                              # This file
```

---

## Setup & Usage

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd student-attendance-risk-prediction
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook YashPatade_student_attendance_risk_prediction.ipynb
```

Select **Kernel → Restart & Run All** to execute every cell from top to bottom.

---

## Dependencies

```text
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
nbformat>=5.7.0
jupyter>=1.0.0
```

---

## Notebook Contents

The notebook has **11 clearly documented sections**:

| Section | Description |
|---|---|
| 1. Import Libraries | Load all required packages; set `random_state=42` |
| 2. Load & Inspect Dataset | Read CSV, check shape, dtypes, nulls, class balance |
| 3. Exploratory Data Analysis | Class distribution pie/bar, KDE density plots, categorical attendance rates, correlation heatmap, box plots |
| 4. Data Preprocessing | Drop `student_id` & `absence_reason`, label-encode 8 categorical columns |
| 5. Train–Test Split | 80/20 stratified split; preserves class ratio |
| 6. Model Training — Decision Tree | `DecisionTreeClassifier` with Gini criterion + 5-fold Stratified CV |
| 7. Model Evaluation | Metrics summary, classification report, confusion matrix, prediction breakdown bar, ROC curve, metrics bar chart |
| 8. Feature Importance | Feature importance chart + ranked text output |
| 9. Sample Predictions | 10 test-set predictions with Risk Level and ✓/✗ indicator |
| 10. Conclusions | Findings, rationale, recommendations, future work |
| 11. Live Student Attendance Risk Predictor | Enter student details and generate attendance prediction and risk level |

---

## Methodology

### Preprocessing
- Dropped `student_id` — no predictive value (arbitrary identifier).
- Dropped `absence_reason` — recorded **after** the absence (data leakage).
- Applied `LabelEncoder` to 8 categorical columns: `gender`, `course`, `year`, `parent_education`, `internet_access`, `hostel_resident`, `class_type`, `weather`.
- No feature scaling needed — Decision Trees are scale-invariant.
- **80/20 stratified train-test split** with `random_state=42`.

### Train-Test Split

| Dataset | Samples |
|---|---:|
| Training | 16,000 |
| Testing | 4,000 |
| Total | 20,000 |

### Algorithm — Decision Tree Classifier

| Hyperparameter | Value | Reason |
|---|---|---|
| `criterion` | `'gini'` | Gini impurity for split quality |
| `max_depth` | `8` | Controls tree complexity |
| `min_samples_split` | `20` | Requires enough data to split a node |
| `min_samples_leaf` | `10` | Ensures leaf nodes are statistically meaningful |
| `class_weight` | `'balanced'` | Accounts for class distribution |
| `random_state` | `42` | Full reproducibility |

**Trained Tree Depth:** 8  
**Leaf Nodes:** 226

**Cross-validation:** 5-fold Stratified K-Fold on the training set.

### Cross-Validation Results

| Fold | Score |
|---|---:|
| Fold 1 | 59.09% |
| Fold 2 | 59.19% |
| Fold 3 | 59.53% |
| Fold 4 | 59.75% |
| Fold 5 | 59.22% |
| **Mean** | **59.36%** |
| **Std** | **0.25%** |

---

## Results

> Results shown below are the actual outputs from the notebook on the 4,000-sample test set.

| Metric | Score |
|---|---:|
| Accuracy | **59.80%** |
| Precision | **60.57%** |
| Recall | **63.39%** |
| F1-Score | **61.95%** |
| ROC-AUC | **61.75%** |
| 5-Fold CV Mean | **59.36%** |

### Correct Predictions

**2,392 / 4,000**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Absent | 59% | 56% | 57% | 1,935 |
| Present | 61% | 63% | 62% | 2,065 |

---

## Top Predictive Features (Decision Tree)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `study_hours` | 0.3427 |
| 2 | `sleep_hours` | 0.2217 |
| 3 | `travel_time_minutes` | 0.1670 |
| 4 | `class_type` | 0.0809 |
| 5 | `weather` | 0.0790 |
| 6 | `age` | 0.0341 |
| 7 | `course` | 0.0194 |
| 8 | `parent_education` | 0.0170 |
| 9 | `year` | 0.0140 |
| 10 | `gender` | 0.0115 |
| 11 | `internet_access` | 0.0079 |
| 12 | `hostel_resident` | 0.0047 |

The three highest feature importance values were:

1. `study_hours` — **0.3427**
2. `sleep_hours` — **0.2217**
3. `travel_time_minutes` — **0.1670**

---

## Sample Predictions

The notebook generates predictions for **10 randomly selected students** from the test dataset.

Each prediction contains:

- Actual Attendance
- Predicted Attendance
- Probability of Present
- Risk Level
- Whether the prediction was correct

### Sample Prediction Risk Levels

The sample prediction section uses:

```text
Probability >= 0.60  → LOW
Probability >= 0.40  → MEDIUM
Otherwise             → HIGH
```

---

## Live Student Attendance Risk Predictor

The notebook also includes a live prediction system where student information can be entered manually.

### Input Features

```text
age
gender
course
year
parent_education
internet_access
hostel_resident
class_type
weather
study_hours
sleep_hours
travel_time_minutes
```

The system provides:

```text
Prediction
Probability of Present
Probability of Absent
Risk Level
Top contributing factors
```

### Live Predictor Risk Levels

The live predictor uses:

```text
Probability >= 0.65  → LOW RISK
Probability >= 0.40  → MEDIUM RISK
Otherwise             → HIGH RISK
```

### Example Output From Notebook

```text
Prediction: PRESENT

Prob(Present): 56.1%
Prob(Absent): 43.9%

Risk Level: MEDIUM RISK
```

---

## Key Findings

- `study_hours` was the **most important feature** in the trained Decision Tree.
- `sleep_hours` was the second most important feature.
- `travel_time_minutes` was the third most important feature.
- `class_type` and `weather` also contributed to the model's predictions.
- The model achieved **59.80% accuracy** on the held-out test dataset.
- The model correctly predicted **2,392 out of 4,000** test samples.
- Feature importance values describe how the trained model uses the features and should not be interpreted as causal relationships.

---

## Reproducibility

All code uses `random_state=42`. The full pipeline — from raw CSV to predictions — runs end-to-end in the notebook with no external dependencies beyond `requirements.txt`.

---

## Report

`projectReport.docx` contains the complete internship report:

- Abstract
- Introduction & Objectives
- Dataset description with feature table
- Methodology (preprocessing + Decision Tree rationale)
- EDA findings
- Model description & hyperparameters
- Results table & feature importance
- Conclusion & future scope
- References

---

## Future Work

- Hyperparameter tuning with `GridSearchCV`
- Class imbalance handling with SMOTE
- Compare with Random Forest / Gradient Boosting
- Deploy as a REST API (FastAPI / Flask)
- Integrate with a live student management dashboard

---

**Author:** Yash Patade | IBM SkillsBuild Internship
