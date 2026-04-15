# 🏥 MEDOPT — Clinical Intelligent Screening System

> **Meta-Heuristic Optimisation (MHO) for Disease Prediction**  
> Applies GWO, PSO, and WOA to solve three real-world clinical problems across three disease domains via an interactive Streamlit web application.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Diseases & Datasets](#2-diseases--datasets)
3. [System Architecture](#3-system-architecture)
4. [The Three Problems](#4-the-three-problems)
5. [Optimisation Algorithms](#5-optimisation-algorithms)
6. [Hyperparameter Tuning (Problem 4)](#6-hyperparameter-tuning-problem-4)
7. [Full Workflow](#7-full-workflow)
8. [Project Structure](#8-project-structure)
9. [Installation & Running Steps](#9-installation--running-steps)
10. [Using the Application](#10-using-the-application)
11. [Technical Details](#11-technical-details)

---

## 1. Project Overview

MEDOPT is a multi-disease clinical decision support system that uses **bio-inspired (meta-heuristic) optimisation algorithms** to:

- Select only the most diagnostically relevant features (tests) for a patient
- Optimise the decision threshold to reduce missed diagnoses
- Auto-tune a Random Forest classifier for maximum accuracy
- Provide explainable, risk-scored patient diagnoses through a web UI

The system is built on the idea that clinical screening does not require every possible test — and that an optimiser can figure out exactly which tests matter most, where to draw the sick/healthy line, and how the model itself should be configured.

---

## 2. Diseases & Datasets

| Disease | Dataset | Samples | Features | Target Column |
|---|---|---|---|---|
| 🧠 Parkinson's Disease | UCI Parkinson's Voice Dataset | 195 | 22 | `status` |
| 💉 Diabetes | PIMA Indians Diabetes Dataset | 768 | 8 | `Outcome` |
| ❤️ Heart Disease | Cleveland Heart Disease Dataset | 303 | 13 | `condition` |

### Parkinson's Disease Features
Voice biomarkers extracted from sustained phonation recordings:
`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`, `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`

### Diabetes Features
Standard PIMA clinical panel:
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### Heart Disease Features
Cleveland cardiac panel:
`age`, `sex`, `cp` (chest pain type), `trestbps` (resting BP), `chol`, `fbs`, `restecg`, `thalach` (max heart rate), `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        app.py  (Streamlit UI)               │
│  Overview │ Problem 1 │ Problem 2 │ Problem 4 │ Diagnosis   │
└──────────────────────────┬─────────────────────────────────┘
                           │
                      main.py  (dispatcher)
                     /    |     \
          run_problem1  run_problem2  run_problem4
               │              │              │
    ┌──────────┴──────┐  ┌────┴────┐  ┌─────┴──────┐
    │  optimization/  │  │problems/│  │  problems/ │
    │  gwo.py         │  │problem2 │  │  problem4  │
    │  pso.py         │  │_thresh  │  │  _hyperparam│
    │  woa.py         │  │_opt.py  │  │  _opt.py   │
    └──────┬──────────┘  └────┬────┘  └─────┬──────┘
           │                  │              │
    models/train_model.py     │              │
    (RandomForestClassifier)  │              │
           │                  │              │
    utils/preprocessing.py ───┴──────────────┘
    utils/visualization.py
    utils/explainability.py
    data_configs.py
```

---

## 4. The Three Problems

### Problem 1 — Feature Selection (GWO / PSO / WOA)

**Objective:** Find the minimum subset of diagnostic features that achieves the highest classification accuracy.

**How it works:**
- Each "agent" (wolf / particle / whale) represents a binary feature mask (1 = include, 0 = exclude)
- The **fitness function** trains an SVC (Support Vector Machine with RBF kernel) on the selected features and evaluates accuracy on the test set
- The algorithm runs for N iterations, progressively converging on the feature combination with the best accuracy
- The final best mask is used to train a **Random Forest** classifier (the main production model)

**Baseline vs Optimised:**
- Baseline = Random Forest trained on ALL features
- Optimised = Random Forest trained on only the selected features

**Output:**
- Best feature subset (selected indices + names)
- Baseline accuracy vs optimised accuracy (+ delta)
- Test reduction percentage
- Convergence curve (fitness vs iteration)
- Confusion matrices (baseline vs optimised)
- Classification report
- Feature correlation heatmap

---

### Problem 2 — Clinical Threshold Optimisation (PSO)

**Objective:** Find the decision threshold that maximises clinical sensitivity (fewer missed diagnoses) while keeping specificity acceptable.

**The Clinical Problem:**
Standard classifiers use a fixed threshold of 0.5 — if the predicted probability > 0.5, the patient is classified as sick. But in medicine, a False Negative (sick patient missed) is far more harmful than a False Positive (healthy patient flagged for extra tests). The optimal threshold should be **lower than 0.5** in most clinical scenarios.

**Fitness Function:**
```
Fitness = α × Sensitivity + (1 - α) × Specificity

Where:
  Sensitivity = TP / (TP + FN)   [proportion of sick patients correctly caught]
  Specificity = TN / (TN + FP)   [proportion of healthy patients correctly excluded]
  α = sensitivity weight (user-controllable, default 0.75)
  
  α → 1.0 : maximise catching sick patients (high recall)
  α → 0.5 : balanced clinical trade-off
```

**PSO Search Space:**
- 1-dimensional: threshold ∈ [0.10, 0.90]
- 20 particles, 30 iterations

**Output:**
- PSO-optimised threshold value
- Default (0.5) vs optimal sensitivity and specificity
- Reduction in False Negatives (missed patients)
- ROC curve with optimal threshold marked
- Threshold analysis plot (sensitivity/specificity vs threshold)
- Confusion matrices (default vs optimal threshold)

---

### Problem 4 — Hyperparameter Tuning (PSO)

**Objective:** Auto-tune the Random Forest classifier's hyperparameters on the feature-selected data from Problem 1, using 5-fold cross-validation as the quality measure.

> Full details in the dedicated section below.

---

## 5. Optimisation Algorithms

All three algorithms are **population-based, iterative, nature-inspired** optimisers. They start with a random population and iteratively update positions guided by the best solutions found so far.

---

### 5.1 Grey Wolf Optimiser (GWO)

**Inspired by:** Grey wolf pack hunting hierarchy and social behaviour (Mirjalili et al., 2014)

**Key Concepts:**
- **Alpha (α)** — the best solution found so far (pack leader)
- **Beta (β)** — the second best solution
- **Delta (δ)** — the third best solution
- **Omega wolves** — the rest of the pack; they update positions based on α, β, δ

**Position Update Rule:**
```
a = 2 - t × (2 / max_iter)    [linearly decreases from 2 → 0]

For each wolf i and feature dimension j:
  X1 = α[j] - A1 × |C1 × α[j] - wolf[i][j]|
  X2 = β[j] - A2 × |C2 × β[j] - wolf[i][j]|
  X3 = δ[j] - A3 × |C3 × δ[j] - wolf[i][j]|

  wolf[i][j] = (X1 + X2 + X3) / 3

Where A = 2a·r1 - a,   C = 2·r2   (r1, r2 ~ Uniform[0,1])
```

**Behaviour:**
- When |A| > 1 → exploration (move away from prey)
- When |A| < 1 → exploitation (encircle and attack prey)
- The decreasing `a` coefficient transitions the algorithm from exploration to exploitation over iterations

**Fitness:** SVC (RBF kernel) accuracy on selected features

---

### 5.2 Particle Swarm Optimiser (PSO)

**Inspired by:** Flocking behaviour of birds and schooling of fish (Kennedy & Eberhart, 1995)

**Key Concepts:**
- Each **particle** has a position (feature mask) and a **velocity**
- **pbest** — the best position each particle has personally visited
- **gbest** — the best position any particle in the swarm has ever visited

**Velocity & Position Update Rule:**
```
For each particle i and feature j:
  v[i] = w × v[i]
        + c1 × r1[j] × (pbest[i][j] - x[i][j])    [cognitive: pull toward own best]
        + c2 × r2[j] × (gbest[j]    - x[i][j])    [social: pull toward swarm best]
  
  x[i][j] = x[i][j] + v[i][j]
  x[i][j] = clip(x[i][j], 0, 1)

Hyperparameters:
  w  = 0.5   (inertia weight — momentum)
  c1 = 1.5   (cognitive coefficient)
  c2 = 1.5   (social coefficient)
  r1, r2 ~ Uniform[0,1] per feature (vectorised)
```

**Used in three places:**
1. Problem 1 — Feature selection (binary mask, n_features dimensional)
2. Problem 2 — Threshold optimisation (1-dimensional, search space [0.10, 0.90])
3. Problem 4 — Hyperparameter tuning (4-dimensional, normalised [0,1])

---

### 5.3 Whale Optimisation Algorithm (WOA)

**Inspired by:** Humpback whale bubble-net hunting strategy (Mirjalili & Lewis, 2016)

**Key Concepts:**
- Whales use two mechanisms: **encircling prey** (exploitation) and **bubble-net spiralling**
- Exploration is achieved by moving toward a **random whale** instead of the best

**Position Update Rule:**
```
a = 2 - t × (2 / max_iter)    [decreases 2 → 0]
A = 2a·r - a   (r ~ Uniform[0,1] per dimension)
C = 2·r
p ~ Uniform[0,1]

If p < 0.5:
  If |A| < 1:  # Exploitation — shrinking encirclement
    D = |C × best - whale[i]|
    whale[i] = best - A × D
  Else:        # Exploration — move toward random whale
    D = |C × X_rand - whale[i]|
    whale[i] = X_rand - A × D

Else (p ≥ 0.5):  # Spiral bubble-net update
  D' = |best - whale[i]|
  l ~ Uniform[-1, 1]
  whale[i] = D' × exp(b × l) × cos(2π × l) + best

b = 1.0  (spiral shape constant)
```

**Behaviour:**
- The spiral update mimics the helical path whales take when herding prey in bubble-net attacks
- The probability `p` controls whether a whale exploits (encircles) or does the spiral

---

### Algorithm Comparison Summary

| Property | GWO | PSO | WOA |
|---|---|---|---|
| Inspired by | Wolf pack hierarchy | Bird flocking | Whale bubble-net hunting |
| Leadership | Alpha/Beta/Delta wolves | Global best (gbest) | Single global best |
| Exploration | |A| > 1 (diverge from prey) | c1, c2 balance | Random whale movement |
| Exploitation | |A| < 1 (encircle prey) | Pull toward pbest/gbest | Shrinking encirclement + spiral |
| Key parameter | a (decreases 2→0) | w, c1, c2 | a, b, p |
| Used for | Problem 1 | P1, P2, P4 | Problem 1 |

---

## 6. Hyperparameter Tuning (Problem 4)

### What Gets Tuned

The **Random Forest Classifier** has 4 hyperparameters being optimised:

| Hyperparameter | Default (sklearn) | Search Range | Meaning |
|---|---|---|---|
| `n_estimators` | 100 | 50 – 300 | Number of decision trees in the forest |
| `max_depth` | None (unlimited) | 3 – 25 | Maximum depth of each tree |
| `min_samples_split` | 2 | 2 – 20 | Min samples needed to split a node |
| `min_samples_leaf` | 1 | 1 – 10 | Min samples required at each leaf node |

### How PSO Tunes These

The PSO operates in a **4-dimensional normalised space** [0, 1]⁴. Each particle's position is a 4-element vector in [0,1], which gets decoded into actual hyperparameter values:

```python
decoded_value = int(round(low + particle[i] * (high - low)))

# Example: particle = [0.3, 0.7, 0.1, 0.5]
#   n_estimators      = round(50  + 0.3 × (300-50))  = round(125)  = 125
#   max_depth         = round(3   + 0.7 × (25-3))    = round(18.4) = 18
#   min_samples_split = round(2   + 0.1 × (20-2))    = round(3.8)  = 4
#   min_samples_leaf  = round(1   + 0.5 × (10-1))    = round(5.5)  = 6
```

### Fitness Function

```python
fitness = mean cross-validation accuracy (5-fold, on training set only)

# For each particle position:
#   1. Decode hyperparameters
#   2. Create RandomForestClassifier with those params
#   3. Run 5-fold cross-validation on X_train[:, selected_features]
#   4. Return mean accuracy across 5 folds
```

This ensures the tuned hyperparameters **generalise** — they are evaluated on unseen folds, not the full training set. This prevents overfitting to the training data.

### PSO Configuration for Problem 4

```
n_particles = 15
max_iter    = 20 (fast mode: 15)
cv_folds    = 5
w  = 0.5   (inertia)
c1 = 1.5   (cognitive)
c2 = 1.5   (social)
```

### Output

- Best hyperparameter set (decoded dictionary)
- Default RF accuracy vs PSO-tuned RF accuracy
- Best cross-validation score
- Convergence curve (CV accuracy vs iteration)
- Parameter comparison chart (default vs tuned)
- Tuned model confusion matrix + classification report

---

## 7. Full Workflow

This is the complete end-to-end pipeline when you run the application:

```
Step 1: DATA LOADING & PREPROCESSING
  ├─ Read CSV from data/ folder
  ├─ Drop identifier columns (e.g. 'name' for Parkinson's)
  ├─ Remove duplicate rows
  ├─ Impute missing values with column medians
  ├─ Split features (X) and target (y)
  ├─ StandardScaler → zero mean, unit variance
  └─ 80/20 stratified train/test split (random_state=42)

Step 2: BASELINE (Problem 1 comparison reference)
  ├─ Train RandomForest on ALL features
  └─ Record baseline accuracy, confusion matrix, classification report

Step 3: PROBLEM 1 — FEATURE SELECTION
  ├─ Choose algorithm: GWO, PSO, or WOA
  ├─ Each agent = binary feature mask of length n_features
  ├─ Fitness = SVC (RBF kernel) accuracy on selected features
  ├─ Run for max_iter iterations (Fast: 5 agents × 10 iters)
  ├─ Best binary mask → extract selected feature indices
  ├─ Train final RandomForest on SELECTED features only
  └─ Store: model, selected_indices, accuracy, confusion matrix

Step 4: PROBLEM 2 — THRESHOLD OPTIMISATION
  ├─ Uses the model from Step 3
  ├─ Get prediction probabilities: model.predict_proba(X_test)[:, 1]
  ├─ PSO searches threshold ∈ [0.10, 0.90]
  ├─ Fitness = α×Sensitivity + (1-α)×Specificity
  ├─ 20 particles × 30 iterations
  └─ Returns optimal threshold + clinical impact metrics

Step 5: PROBLEM 4 — HYPERPARAMETER TUNING
  ├─ Uses selected_features from Step 3
  ├─ PSO in 4D normalised space [0,1]^4
  ├─ Fitness = 5-fold CV accuracy on X_train[:, selected]
  ├─ 15 particles × 20 iterations
  ├─ Decode best position → RF hyperparameters
  └─ Train final tuned RandomForest model

Step 6: PATIENT DIAGNOSIS
  ├─ Enter patient values (manual or from test set)
  ├─ StandardScale using training set statistics
  ├─ Select only the features from Step 3
  ├─ Use tuned model (Step 5) or base model (Step 3)
  ├─ Apply optimal threshold from Step 4 (or 0.5)
  ├─ Compute risk score = probability × 100
  ├─ Assign warning level (Low / Moderate / High)
  └─ Explain prediction via SHAP or RF feature importance

Step 7: CROSS-DISEASE COMPARISON
  └─ Table of all Problem 1 results across diseases and algorithms
```

---

## 8. Project Structure

```
MHO/
│
├── app.py                          # Streamlit multi-page application (UI entry point)
├── main.py                         # Business logic dispatcher (run_problem1/2/4)
├── data_configs.py                 # Disease configs: file paths, targets, class names
├── requirements.txt                # Python dependencies
├── PROJECT_OVERVIEW.txt            # Plain-text project explanation
├── README.md                       # This file
│
├── data/
│   ├── parkinsons.csv              # Parkinson's voice dataset (195 × 23)
│   ├── diabetes.csv                # PIMA diabetes dataset (768 × 9)
│   ├── heart_cleveland_upload.csv  # Cleveland heart dataset (303 × 14)
│   └── download_datasets.py        # Script to download datasets
│
├── optimization/
│   ├── gwo.py                      # Grey Wolf Optimiser (feature selection)
│   ├── pso.py                      # Particle Swarm Optimiser (feature selection)
│   └── woa.py                      # Whale Optimisation Algorithm (feature selection)
│
├── problems/
│   ├── problem2_threshold_opt.py   # PSO-based clinical threshold optimisation
│   └── problem4_hyperparam_opt.py  # PSO-based RF hyperparameter tuning
│
├── models/
│   └── train_model.py              # RandomForest training + evaluation helper
│
├── utils/
│   ├── preprocessing.py            # Data loading, cleaning, scaling, splitting
│   ├── visualization.py            # All chart/plot functions (matplotlib)
│   ├── explainability.py           # SHAP explanation + RF importance fallback
│   └── cost_analysis.py            # (Legacy) test panel analysis utilities
│
└── evaluation/
    └── __init__.py                 # Evaluation module placeholder
```

---

## 9. Installation & Running Steps

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- The three dataset CSV files in the `data/` folder

### Step 1 — Clone or Download the Project

```bash
git clone <repository-url>
cd MHO
```

Or simply navigate to the project folder in your terminal.

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install streamlit scikit-learn numpy pandas matplotlib
```

For AI-powered SHAP explanations (optional but recommended):
```bash
pip install shap
```

Full install in one command:
```bash
pip install streamlit scikit-learn numpy pandas matplotlib shap
```

### Step 4 — Verify Datasets

Make sure the following files exist in the `data/` folder:

```
data/parkinsons.csv
data/diabetes.csv
data/heart_cleveland_upload.csv
```

If they are missing, run:
```bash
python data/download_datasets.py
```

### Step 5 — Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at:
```
http://localhost:8501
```

If it doesn't open automatically, navigate to that URL manually.

---

## 10. Using the Application

### Sidebar Controls
- **Disease** — Select the disease to analyse (Parkinson's / Diabetes / Heart Disease)
- **Algorithm** — Choose the MHO algorithm for Problem 1 (GWO / PSO / WOA)
- **Fast Mode** — When checked: 5 agents × 10 iterations (quick). Uncheck for 10 × 20 (thorough)

### Recommended Workflow (Page by Page)

1. **🏠 Overview**
   - Verify all three datasets are shown as ready (🟢)
   - View the feature correlation heatmap for the selected disease

2. **🔬 Problem 1 · Feature Selection**
   - Select your disease and algorithm in the sidebar
   - Click **"Run [Algorithm] Feature Selection"**
   - View selected features, accuracy improvement, convergence curve

3. **🎯 Problem 2 · Threshold Optimisation**
   - Requires Problem 1 to be run first
   - Adjust the `α` sensitivity weight if needed
   - Click **"Run PSO Threshold Optimisation"**
   - Compare default vs optimal threshold clinical impact

4. **⚙️ Problem 4 · Hyperparameter Tuning**
   - Requires Problem 1 to be run first
   - Click **"Run PSO Hyperparameter Tuning"**
   - Compare default Random Forest vs PSO-tuned RF
   - View best hyperparameter configuration

5. **🩺 Patient Diagnosis**
   - Requires Problem 1 to be run first
   - Choose a patient from the test set or enter values manually
   - Click **"Diagnose Patient"**
   - View risk score, warning level, and feature explanation

6. **📊 Cross-Disease Comparison**
   - Run Problem 1 for multiple disease/algorithm combinations
   - View a consolidated comparison table and accuracy bar chart

---

## 11. Technical Details

### Data Preprocessing Pipeline

```python
# 1. Load CSV
df = pd.read_csv(path)

# 2. Drop non-predictive columns (e.g. patient name)
df = df.drop(columns=drop_cols)

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Impute missing values with column medians
for col in numeric_columns:
    df[col].fillna(df[col].median())

# 5. Standardise features
X_scaled = StandardScaler().fit_transform(X)

# 6. Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
```

### Fitness Function (Problems 1 — Feature Selection)

```python
def fitness(feature_mask, X_train, X_test, y_train, y_test):
    selected = np.where(feature_mask > 0.5)[0]
    if len(selected) == 0:
        return 0.0
    model = SVC(kernel="rbf")
    model.fit(X_train[:, selected], y_train)
    return accuracy_score(y_test, model.predict(X_test[:, selected]))
```

**Why SVC for fitness and RandomForest for the final model?**  
SVC is fast to train and computationally efficient for the many evaluations required during optimisation (hundreds of fitness calls per run). Once the best features are found, a RandomForest is trained as the final model because it is interpretable, probabilistic (needed for thresholding and SHAP), and robust.

### Fast Mode vs Full Mode

| Setting | Agents | Iterations | Approx. Time |
|---|---|---|---|
| Fast Mode ✅ | 5 | 10 | ~30 seconds |
| Full Mode | 10 | 20 | ~2–5 minutes |

### Reproducibility

All algorithms use a seeded NumPy random generator (`np.random.default_rng(seed=42)`) so results are fully reproducible across runs. The train/test split also uses `random_state=42`.

### SHAP Explainability

When SHAP is installed, the Patient Diagnosis page uses **TreeExplainer** to compute Shapley values for the Random Forest prediction. These show exactly how much each feature contributed (positively or negatively) to the individual patient's risk score.

If SHAP is not installed, the system falls back to using the built-in RandomForest **feature_importances_** (mean impurity decrease) as a less granular but still useful explanation.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.28 | Web application framework |
| `scikit-learn` | ≥1.3 | SVC, RandomForest, metrics, preprocessing |
| `numpy` | ≥1.24 | Numerical arrays and RNG |
| `pandas` | ≥2.0 | DataFrame handling and CSV loading |
| `matplotlib` | ≥3.7 | All charts and plots |
| `shap` | ≥0.43 | AI explainability (optional) |

---

*MEDOPT — Built with Meta-Heuristic Optimisation for smarter clinical screening.*
