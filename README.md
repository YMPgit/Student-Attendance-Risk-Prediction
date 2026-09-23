# Student Attendance Risk Prediction

> **IBM SkillsBuild Internship Project**
> Author: Yash Patade | Algorithm: Decision Tree Classifier

A beginner-friendly, end-to-end machine learning project that predicts whether a student will **attend** (`1`) or be **absent** (`0`) from class, built with a Decision Tree classifier on a real Kaggle dataset. Clean, reproducible, and suitable for an IBM internship submission.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Student Attendance Prediction Dataset |
| **Source** | [Kaggle — kundanbedmutha/student-attendance-dataset-college-level](https://www.kaggle.com/datasets/kundanbedmutha/student-attendance-dataset-college-level) |
| **File** | `Attendance_Prediction.csv` |
| **Records** | 20,000 rows × 15 columns |
| **Target** | `attendance` — binary (1 = Present, 0 = Absent) |

---

## Project Structure

```
.
├── student_attendance_prediction.ipynb    # Main Jupyter notebook (EDA → Training → Results)
├── Attendance_Prediction.csv              # Real dataset from Kaggle
├── requirements.txt                       # Python dependencies
├── projectReport.docx                     # Full internship project report
└── README.md                              # This file
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
jupyter notebook student_attendance_prediction.ipynb
```

Select **Kernel → Restart & Run All** to execute every cell from top to bottom.

---

## Dependencies

```
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

The notebook has **10 clearly documented sections**:

| Section | Description |
|---|---|
| 1. Import Libraries | Load all required packages; set `random_state=42` |
| 2. Load & Inspect Dataset | Read CSV, check shape, dtypes, nulls, class balance |
| 3. Exploratory Data Analysis | Class distribution pie/bar, KDE density plots, categorical attendance rates, correlation heatmap, box plots |
| 4. Data Preprocessing | Drop `student_id` & `absence_reason`, label-encode 8 categorical columns |
| 5. Train–Test Split | 80/20 stratified split; preserves class ratio |
| 6. Model Training — Decision Tree | `DecisionTreeClassifier` with Gini criterion + 5-fold Stratified CV |
| 7. Model Evaluation | Metrics summary, classification report, confusion matrix, prediction breakdown bar, ROC curve, metrics bar chart |
| 8. Feature Importance | Colour-coded horizontal bar chart + ranked text output |
| 9. Sample Predictions | 10 test-set predictions with Risk Level (Low/Medium/High) and ✓/✗ indicator |
| 10. Conclusions | Findings, rationale, recommendations, future work |

---

## Methodology

### Preprocessing
- Dropped `student_id` — no predictive value (arbitrary identifier).
- Dropped `absence_reason` — recorded **after** the absence (data leakage).
- Applied `LabelEncoder` to 8 categorical columns: `gender`, `course`, `year`, `parent_education`, `internet_access`, `hostel_resident`, `class_type`, `weather`.
- No feature scaling needed — Decision Trees are scale-invariant.
- **80/20 stratified train-test split** with `random_state=42`.

### Algorithm — Decision Tree Classifier

| Hyperparameter | Value | Reason |
|---|---|---|
| `criterion` | `'gini'` | Gini impurity for split quality |
| `max_depth` | `8` | Prevents overfitting |
| `min_samples_split` | `20` | Requires enough data to split a node |
| `min_samples_leaf` | `10` | Ensures leaf nodes are statistically meaningful |
| `class_weight` | `'balanced'` | Accounts for slight class imbalance |
| `random_state` | `42` | Full reproducibility |

**Cross-validation:** 5-fold Stratified K-Fold on the training set.

---

## Results

> Run the notebook for exact scores. Approximate values shown below.

| Metric | Score |
|---|---|
| Accuracy | ~0.68 |
| Precision | ~0.70 |
| Recall | ~0.72 |
| F1-Score | ~0.71 |
| ROC-AUC | ~0.68 |
| 5-Fold CV Mean | ~0.67 |

### Top Predictive Features (Decision Tree)
1. `study_hours` — strongest single predictor of attendance
2. `travel_time_minutes` — longer commutes increase absence risk
3. `sleep_hours` — sleep quality and quantity matter
4. `internet_access` — online access supports attendance
5. `hostel_resident` — on-campus students avoid commute barriers

---

## Key Findings

- Students who **study ≥ 3 hours/day** are significantly more likely to attend.
- **Travel time > 60 minutes** is the strongest environmental risk factor.
- Students with **internet access** and **on-campus housing** attend more regularly.
- **Rainy / cold weather** correlates with higher absenteeism.
- No missing values — the dataset is clean and immediately usable.

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
