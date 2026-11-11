# Copilot instructions (DavidEkpechue / diabetes classifier)

## Project shape
- This repo is a **single-script ML analysis** driven by `diabetes_classifier.py` using `diabetes.csv` (Pima Indians Diabetes Database).
- The script is intentionally narrative/sectioned (prints `SECTION 1..6`) and produces a fixed set of `.png` artifacts referenced by `README_DIABETES_CLASSIFIER.md` and `DOCUMENTATION.md`.

## How to run (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\diabetes_classifier.py
```
- Run from the repo root so relative paths resolve: `pd.read_csv('diabetes.csv')` and `plt.savefig('*.png')`.

## Data flow & conventions (match existing behavior)
- **Missing-value convention:** zeros in `['Glucose','BloodPressure','SkinThickness','Insulin','BMI']` are treated as missing and replaced with `NaN`, then median-imputed via `SimpleImputer`.
- **Reproducibility:** keep `RANDOM_SEED = 42`, `np.random.seed(RANDOM_SEED)`, and pass `random_state=RANDOM_SEED` to models/split.
- **Scaling:** `StandardScaler` is applied and the *scaled* arrays (`X_train_scaled`, `X_test_scaled`) are used for both Logistic Regression and Random Forest for consistency.
- **Modeling pattern:** train baseline → `cross_val_score(..., cv=5, scoring='accuracy')` → `GridSearchCV(..., cv=5, scoring='accuracy')` → final evaluation.
- **Evaluation pattern:** use `evaluate_model(...)` which expects `predict_proba` and returns metrics + predictions + confusion matrix.

## Output artifacts (keep filenames stable)
Running `diabetes_classifier.py` writes these images in the repo root:
- `data_distributions.png`, `correlation_heatmap.png`, `outlier_boxplots.png`
- `confusion_matrices.png`, `roc_curves.png`, `metrics_comparison.png`, `feature_importance.png`

## Making changes safely
- Prefer **minimal diffs** in `diabetes_classifier.py`; keep section headers/numbering and the printed “ANALYSIS COMPLETE” summary intact.
- If you change preprocessing/modeling (e.g., move imputation/scaling into a pipeline), update the documentation pointers in `README_DIABETES_CLASSIFIER.md` / `DOCUMENTATION.md` and verify the expected `.png` outputs still generate.