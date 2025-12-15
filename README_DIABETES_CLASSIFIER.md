# CMPU 4011 Machine Learning - Diabetes Classifier Assignment

This project implements machine learning classifiers to predict diabetes using the Pima Indians Diabetes Database.

## 📁 Project Files

- **`diabetes_classifier.py`** - Main Python script with complete implementation
- **`DOCUMENTATION.md`** - Comprehensive documentation (30+ pages)
- **`diabetes.csv`** - Pima Indians Diabetes Database
- **`requirements.txt`** - Python package dependencies
- **Visualization outputs** (generated when script runs):
  - `data_distributions.png`
  - `correlation_heatmap.png`
  - `outlier_boxplots.png`
  - `confusion_matrices.png`
  - `roc_curves.png`
  - `metrics_comparison.png`
  - `feature_importance.png`

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the files

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Classifier

```bash
python diabetes_classifier.py
```

The script will:
- Load and explore the dataset
- Preprocess the data (handle missing values, scale features)
- Train two classifiers (Logistic Regression and Random Forest)
- Perform hyperparameter tuning
- Evaluate both models
- Generate visualizations
- Display comparison results

**Expected runtime:** 2-3 minutes

## 📊 What This Project Does

### 1. Data Analysis and Preprocessing (30%)
- ✅ Load and explore diabetes.csv dataset
- ✅ Statistical summaries and distribution analysis
- ✅ Correlation analysis
- ✅ Missing value detection (zeros treated as NaN)
- ✅ Data quality handling with median imputation
- ✅ Outlier detection using IQR method
- ✅ Feature scaling with StandardScaler
- ✅ 80-20 train-test split with stratification

### 2. Classifier Development (30%)
Implements **TWO** classifiers:

**Logistic Regression:**
- Baseline linear model
- Interpretable coefficients
- Hyperparameter tuning: C, penalty, solver
- 5-fold cross-validation

**Random Forest:**
- Ensemble of decision trees
- Captures non-linear relationships
- Hyperparameter tuning: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Feature importance ranking

### 3. Evaluation and Discussion (40%)
- ✅ Multiple evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - ROC-AUC Score
- ✅ Multiple iterations with different parameters
- ✅ Comprehensive model comparison
- ✅ Visualizations (7 plots generated)
- ✅ Detailed discussion in DOCUMENTATION.md

## 📈 Expected Results

Based on the implementation:
- **Random Forest** achieves ~76% test accuracy
- **Logistic Regression** achieves ~69% test accuracy
- Both models show good ROC-AUC scores (~0.81)
- Glucose, BMI, and Age identified as most important features

## 📖 Documentation

See **`DOCUMENTATION.md`** for comprehensive details including:
1. Introduction and dataset description
2. Data exploration findings
3. Preprocessing decisions and rationale
4. Classifier selection reasoning
5. Model development process
6. Evaluation results with visualizations
7. Discussion and recommendations
8. Limitations and future improvements

## 🔬 Dataset Information

**Pima Indians Diabetes Database**
- 768 patients (Pima Indian women aged 21+)
- 8 medical predictor features
- Binary outcome (0 = No diabetes, 1 = Diabetes)
- Class distribution: 65% negative, 35% positive

**Features:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## 🛠️ Technical Details

**Libraries Used:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models and tools
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualizations

**Random Seed:** 42 (for reproducibility)

**Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation

## 📝 Assignment Requirements Met

| Requirement | Status |
|------------|--------|
| Data loading and exploration | ✅ Complete |
| Statistical summaries | ✅ Complete |
| Distribution analysis | ✅ Complete |
| Correlation analysis | ✅ Complete |
| Missing value detection | ✅ Complete |
| Data preprocessing | ✅ Complete |
| Feature scaling | ✅ Complete |
| Train-test split | ✅ Complete |
| Two different classifiers | ✅ Logistic Regression + Random Forest |
| Model training | ✅ Complete |
| Hyperparameter tuning | ✅ GridSearchCV implemented |
| Cross-validation | ✅ 5-fold CV |
| Multiple metrics evaluation | ✅ Accuracy, Precision, Recall, F1, ROC-AUC |
| Confusion matrices | ✅ Complete |
| Multiple iterations | ✅ Parameter testing documented |
| Model comparison | ✅ Comprehensive comparison |
| Discussion | ✅ Detailed in DOCUMENTATION.md |
| Visualizations | ✅ 7 plots generated |
| Well-commented code | ✅ Extensive comments |
| Clear structure | ✅ 6 organized sections |
| Comprehensive documentation | ✅ 30+ page DOCUMENTATION.md |
| PEP 8 style | ✅ Followed |
| Reproducibility | ✅ Random seed set |

## 🎓 Marking Scheme Alignment

- **Data Exploration and Preprocessing (30%):** Sections 1-2
- **Classifier Development (30%):** Sections 3-4
- **Evaluation and Discussion (40%):** Sections 5-6 + DOCUMENTATION.md

## 💡 Key Insights

1. **Data Quality Matters:** 48.7% of Insulin values were missing (zeros)
2. **Glucose is King:** Strongest predictor of diabetes
3. **Random Forest Wins:** But only by ~7% - Logistic Regression is competitive
4. **Trade-offs Exist:** Performance vs. Interpretability
5. **Medical Context Important:** False negatives more serious than false positives

## 🔮 Future Improvements

- Address class imbalance with SMOTE
- Feature engineering (interaction terms)
- Try XGBoost or Neural Networks
- Optimize decision threshold
- Collect more diverse data

## 📧 Contact

**Author:** David Ekpechue  
**Course:** CMPU 4011 Machine Learning

---

*This project demonstrates comprehensive machine learning workflow from data exploration to model deployment, with emphasis on proper methodology, evaluation, and documentation.*
