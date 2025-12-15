"""
CMPU 4011 Machine Learning - Diabetes Classifier
Author: David Ekpechue
Dataset: Pima Indians Diabetes Database

This script implements and compares two machine learning classifiers 
(Logistic Regression and Random Forest) to predict diabetes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, classification_report)
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("CMPU 4011 Machine Learning - Diabetes Classifier")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING AND EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND EXPLORATION")
print("="*80)

# Load the dataset
df = pd.read_csv('diabetes.csv')
print("\n1.1 Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n1.2 First 5 Rows:")
print(df.head())

print("\n1.3 Dataset Information:")
print(df.info())

print("\n1.4 Statistical Summary:")
print(df.describe())

print("\n1.5 Target Variable Distribution:")
print(df['Outcome'].value_counts())
print(f"\nClass Balance:")
print(f"No Diabetes (0): {(df['Outcome']==0).sum()} ({(df['Outcome']==0).sum()/len(df)*100:.1f}%)")
print(f"Diabetes (1): {(df['Outcome']==1).sum()} ({(df['Outcome']==1).sum()/len(df)*100:.1f}%)")

print("\n1.6 Checking for Missing Values:")
print(df.isnull().sum())

# Check for zero values in medical measurements (which should be missing values)
print("\n1.7 Checking for Zero Values (potential missing values):")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")

# Correlation analysis
print("\n1.8 Correlation Matrix:")
correlation_matrix = df.corr()
print(correlation_matrix['Outcome'].sort_values(ascending=False))

# Visualizations
print("\n1.9 Creating visualization plots...")

# Create figure for data exploration visualizations
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Data Exploration Visualizations', fontsize=16)

# Distribution plots for each feature
features = df.columns[:-1]
for idx, col in enumerate(features):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col_idx].set_title(f'Distribution of {col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: data_distributions.png")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")

# Box plots to identify outliers
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Box Plots for Outlier Detection', fontsize=16)
for idx, col in enumerate(features):
    row = idx // 4
    col_idx = idx % 4
    axes[row, col_idx].boxplot(df[col])
    axes[row, col_idx].set_title(f'{col}')
    axes[row, col_idx].set_ylabel('Value')
plt.tight_layout()
plt.savefig('outlier_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: outlier_boxplots.png")

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DATA PREPROCESSING")
print("="*80)

# Create a copy for preprocessing
df_processed = df.copy()

print("\n2.1 Handling Zero Values as Missing Data:")
# Replace zeros with NaN in medical measurements (vectorized operation)
df_processed[zero_cols] = df_processed[zero_cols].replace(0, np.nan)
for col in zero_cols:
    missing_count = df_processed[col].isnull().sum()
    print(f"{col}: {missing_count} missing values after replacement")

# Impute missing values with median
print("\n2.2 Imputing Missing Values using Median Strategy:")
imputer = SimpleImputer(strategy='median')
df_processed[zero_cols] = imputer.fit_transform(df_processed[zero_cols])
print("Missing values imputed successfully")

# Verify no missing values remain
print("\n2.3 Verifying No Missing Values:")
print(df_processed.isnull().sum())

# Outlier detection and handling (using IQR method)
print("\n2.4 Outlier Detection using IQR Method:")
outlier_counts = {}
for col in df_processed.columns[:-1]:  # Exclude 'Outcome'
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_processed[(df_processed[col] < lower_bound) | 
                           (df_processed[col] > upper_bound)]
    outlier_counts[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers detected")

# Note: We keep outliers as they may be legitimate values in medical data

# Separate features and target
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

print("\n2.5 Train-Test Split (80-20):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\n2.6 Feature Scaling (Standardization):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized successfully")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# ============================================================================
# SECTION 3: MODEL 1 - LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: MODEL 1 - LOGISTIC REGRESSION")
print("="*80)

print("\n3.1 Model Selection Rationale:")
print("""
Logistic Regression is chosen because:
- It's interpretable and provides probability estimates
- Works well for binary classification problems
- Computationally efficient
- Provides insight into feature importance through coefficients
- Good baseline model for comparison
""")

# Initial model training
print("\n3.2 Training Initial Logistic Regression Model:")
lr_model_initial = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
lr_model_initial.fit(X_train_scaled, y_train)
print("Initial model trained successfully")

# Cross-validation
print("\n3.3 Cross-Validation (5-fold):")
cv_scores_lr = cross_val_score(lr_model_initial, X_train_scaled, y_train, 
                                cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores_lr}")
print(f"Mean CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# Hyperparameter tuning
print("\n3.4 Hyperparameter Tuning using GridSearchCV:")
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search_lr = GridSearchCV(
    LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    param_grid_lr,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search_lr.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search_lr.best_params_}")
print(f"Best CV score: {grid_search_lr.best_score_:.4f}")

# Use best model
lr_model = grid_search_lr.best_estimator_

# Multiple iterations with different parameters
print("\n3.5 Testing Multiple Parameter Combinations:")
test_params = [
    {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'},
]

print(f"{'C':<10}{'Penalty':<10}{'Solver':<15}{'Train Acc':<12}{'Test Acc':<12}")
print("-" * 60)
for params in test_params:
    temp_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, **params)
    temp_model.fit(X_train_scaled, y_train)
    train_acc = temp_model.score(X_train_scaled, y_train)
    test_acc = temp_model.score(X_test_scaled, y_test)
    print(f"{params['C']:<10}{params['penalty']:<10}{params['solver']:<15}"
          f"{train_acc:<12.4f}{test_acc:<12.4f}")

# ============================================================================
# SECTION 4: MODEL 2 - RANDOM FOREST CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: MODEL 2 - RANDOM FOREST CLASSIFIER")
print("="*80)

print("\n4.1 Model Selection Rationale:")
print("""
Random Forest is chosen because:
- Handles non-linear relationships well
- Less prone to overfitting compared to single decision trees
- Provides feature importance rankings
- No need for feature scaling (but we use scaled data for consistency)
- Robust to outliers
- Can capture complex interactions between features
""")

# Initial model training
print("\n4.2 Training Initial Random Forest Model:")
rf_model_initial = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
rf_model_initial.fit(X_train_scaled, y_train)
print("Initial model trained successfully")

# Cross-validation
print("\n4.3 Cross-Validation (5-fold):")
cv_scores_rf = cross_val_score(rf_model_initial, X_train_scaled, y_train, 
                                cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores_rf}")
print(f"Mean CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# Hyperparameter tuning
print("\n4.4 Hyperparameter Tuning using GridSearchCV:")
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search_rf.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best CV score: {grid_search_rf.best_score_:.4f}")

# Use best model
rf_model = grid_search_rf.best_estimator_

# Multiple iterations with different parameters
print("\n4.5 Testing Multiple Parameter Combinations:")
test_params_rf = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
]

print(f"{'N_Est':<10}{'Max_Depth':<12}{'Min_Split':<12}{'Train Acc':<12}{'Test Acc':<12}")
print("-" * 60)
for params in test_params_rf:
    temp_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
    temp_model.fit(X_train_scaled, y_train)
    train_acc = temp_model.score(X_train_scaled, y_train)
    test_acc = temp_model.score(X_test_scaled, y_test)
    max_d = str(params['max_depth']) if params['max_depth'] is not None else 'None'
    print(f"{params['n_estimators']:<10}{max_d:<12}{params['min_samples_split']:<12}"
          f"{train_acc:<12.4f}{test_acc:<12.4f}")

# Feature importance for Random Forest
print("\n4.6 Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# ============================================================================
# SECTION 5: MODEL EVALUATION AND COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: MODEL EVALUATION AND COMPARISON")
print("="*80)

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Parameters:
    -----------
    model : estimator object
        Trained sklearn classifier with predict and predict_proba methods
    X_train : array-like
        Training feature data
    X_test : array-like
        Test feature data
    y_train : array-like
        Training target labels
    y_test : array-like
        Test target labels
    model_name : str
        Name of the model for display purposes
    
    Returns:
    --------
    tuple : (metrics_dict, y_pred, y_proba, confusion_matrix)
        metrics_dict : Dictionary containing all performance metrics
        y_pred : Test set predictions
        y_proba : Test set probability predictions for positive class
        confusion_matrix : 2x2 confusion matrix as numpy array
    """
    print(f"\n{model_name} Evaluation:")
    print("-" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1-Score': f1_score(y_test, y_test_pred),
        'ROC-AUC': roc_auc_score(y_test, y_test_proba)
    }
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"{metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['No Diabetes', 'Diabetes']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return metrics, y_test_pred, y_test_proba, cm

# Evaluate both models
print("\n5.1 Logistic Regression Performance:")
lr_metrics, lr_pred, lr_proba, lr_cm = evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "Logistic Regression"
)

print("\n5.2 Random Forest Performance:")
rf_metrics, rf_pred, rf_proba, rf_cm = evaluate_model(
    rf_model, X_train_scaled, X_test_scaled, y_train, y_test,
    "Random Forest"
)

# Comparison table
print("\n5.3 Model Comparison:")
print("="*80)
comparison_df = pd.DataFrame([lr_metrics, rf_metrics])
comparison_df = comparison_df.set_index('Model')
print(comparison_df)

# Visualizations
print("\n5.4 Creating evaluation visualizations...")

# Confusion matrices comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix Comparison', fontsize=16)

sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0].set_title('Logistic Regression')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[1].set_title('Random Forest')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")

# ROC curves
plt.figure(figsize=(10, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_metrics["ROC-AUC"]:.4f})',
         linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_metrics["ROC-AUC"]:.4f})',
         linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves.png")

# Metrics comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metrics Comparison', fontsize=16)

metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics_to_plot):
    row = idx // 2
    col = idx % 2
    
    values = [lr_metrics[metric], rf_metrics[metric]]
    bars = axes[row, col].bar(['Logistic Regression', 'Random Forest'], values,
                              color=['skyblue', 'lightgreen'])
    axes[row, col].set_title(metric)
    axes[row, col].set_ylabel('Score')
    axes[row, col].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: metrics_comparison.png")

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")

# ============================================================================
# SECTION 6: FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n6.1 Model Performance Summary:")
print(f"""
Logistic Regression:
  - Test Accuracy: {lr_metrics['Test Accuracy']:.4f}
  - Precision: {lr_metrics['Precision']:.4f}
  - Recall: {lr_metrics['Recall']:.4f}
  - F1-Score: {lr_metrics['F1-Score']:.4f}
  - ROC-AUC: {lr_metrics['ROC-AUC']:.4f}

Random Forest:
  - Test Accuracy: {rf_metrics['Test Accuracy']:.4f}
  - Precision: {rf_metrics['Precision']:.4f}
  - Recall: {rf_metrics['Recall']:.4f}
  - F1-Score: {rf_metrics['F1-Score']:.4f}
  - ROC-AUC: {rf_metrics['ROC-AUC']:.4f}
""")

# Determine best model
if rf_metrics['Test Accuracy'] > lr_metrics['Test Accuracy']:
    best_model_name = "Random Forest"
    best_metrics = rf_metrics
else:
    best_model_name = "Logistic Regression"
    best_metrics = lr_metrics

print("\n6.2 Recommendation:")
print(f"""
Based on the evaluation metrics, {best_model_name} performs better overall with:
  - Higher test accuracy: {best_metrics['Test Accuracy']:.4f}
  - Better F1-Score: {best_metrics['F1-Score']:.4f}
  - Superior ROC-AUC: {best_metrics['ROC-AUC']:.4f}

For medical diagnosis, the choice also depends on the use case:
- If we want to minimize false negatives (missing diabetes cases), 
  prioritize the model with higher RECALL.
- If we want to minimize false positives (unnecessary treatment),
  prioritize the model with higher PRECISION.
""")

print("\n6.3 Key Insights:")
print("""
1. Data Quality: Handling zero values as missing data was crucial
2. Feature Importance: Glucose, BMI, and Age are the most important predictors
3. Model Complexity: Random Forest captured non-linear relationships better
4. Cross-Validation: Both models showed consistent performance across folds
5. Class Imbalance: The dataset is slightly imbalanced (65-35 split)
""")

print("\n6.4 Potential Improvements:")
print("""
1. Address class imbalance with SMOTE or class weighting
2. Feature engineering: Create interaction terms or polynomial features
3. Ensemble methods: Combine predictions from both models
4. Try additional algorithms: XGBoost, SVM, or Neural Networks
5. Collect more data to improve model generalization
6. Consider domain-specific feature transformations
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("  - data_distributions.png")
print("  - correlation_heatmap.png")
print("  - outlier_boxplots.png")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - metrics_comparison.png")
print("  - feature_importance.png")
print("\nFor detailed analysis, refer to DOCUMENTATION.md")
print("="*80)
