# credit_risk_pipeline.py
"""
Predictive Analytics & NLP (Credit Risk Models) - end-to-end pipeline
Tech stack: Python, scikit-learn, xgboost, catboost (optional), imbalanced-learn, shap (optional)
Data: Data.xlsx (expected columns per your Data Dictionary). See file for variable definitions. 
Reference: Data dictionary uploaded by user. :contentReference[oaicite:2]{index=2}
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# optional libs
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# 1) LOAD DATA
# ---------------------------
DATA_PATH = "Data.xlsx"  # adjust if needed

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your Data.xlsx in the script folder.")

df = pd.read_excel(DATA_PATH)
print("Raw data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Quick peek
print(df.head())

# ---------------------------
# 2) BASIC CLEANING & MAPPING
# Use mapping according to Data Dictionary (codes -> readable). See uploaded Data Dictionary. :contentReference[oaicite:3]{index=3}
# ---------------------------

# Example: If your categorical codes are already in the dataset (like 'S11', 'A31' etc.)
# We'll keep them as categorical labels and apply OneHot or Target encoding later.

# Rename target if needed
target_col = "Default_On_Payment"
if target_col not in df.columns:
    # try plausible alternative names
    for c in df.columns:
        if c.lower().strip() in ("default", "default_on_payment", "default_on_payment_1"):
            target_col = c
            break
assert target_col in df.columns, "Target column not found. Please check dataset."

# Remove duplicates and obvious bad records
df = df.drop_duplicates().reset_index(drop=True)
print("After dedup shape:", df.shape)

# Check missingness
missing = df.isna().mean().sort_values(ascending=False)
print("Missing fraction per column:\n", missing)

# ---------------------------
# 3) FEATURE ENGINEERING
# ---------------------------

# Some useful numeric transformations:
# - credit_amount to log scale
# - credit_amount / duration -> monthly_payment_est
# - age buckets
# - interactions (e.g., num_credits * credit_amount)

df['Credit_Amount_log'] = np.log1p(df['Credit_Amount'])
df['Monthly_Payment_Est'] = df['Credit_Amount'] / (df['Duration_in_Months'].replace(0,1))
df['Credit_per_CreditCount'] = df['Credit_Amount'] / (df['Num_Credits'].replace(0,1))
df['Age_bin'] = pd.cut(df['Age'], bins=[18,25,35,45,55,65,100], labels=['18-25','26-35','36-45','46-55','56-65','65+'])

# If some categorical columns encode numbers as strings, preserve them
categorical_cols = [
    'Purpose_Credit_Taken', 'Status_Checking_Accnt', 'Credit_History', 'Job_Status',
    'Years_At_Present_Employment', 'Marital_Status_Gender', 'Other_Debtors_Guarantors',
    'Housing'
]
# keep only those present
categorical_cols = [c for c in categorical_cols if c in df.columns]

numeric_cols = [
    'Credit_Amount', 'Duration_in_Months', 'Years_At_Present_Employment', 'Current_Address_Yrs',
    'Age', 'Num_Credits', 'Num_Dependents', 'Credit_Amount_log', 'Monthly_Payment_Est', 'Credit_per_CreditCount'
]
numeric_cols = [c for c in numeric_cols if c in df.columns]

# If Years_At_Present_Employment is encoded as categories like E72 etc, treat as categorical
if 'Years_At_Present_Employment' in df.columns and df['Years_At_Present_Employment'].dtype == 'O':
    if 'Years_At_Present_Employment' in categorical_cols:
        numeric_cols = [c for c in numeric_cols if c != 'Years_At_Present_Employment']

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# ---------------------------
# 4) SPLIT
# ---------------------------
X = df[numeric_cols + categorical_cols + (['Age_bin'] if 'Age_bin' in df.columns else [])].copy()
y = df[target_col].astype(int)

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train positive rate:", y_train.mean(), "Test positive rate:", y_test.mean())

# ---------------------------
# 5) PREPROCESSING PIPELINE
# ---------------------------

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer
# For high-cardinality features, target encoding would be preferred (not included by default).
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [c for c in numeric_cols if c in X_train.columns]),
        ('cat', categorical_transformer, [c for c in categorical_cols + (['Age_bin'] if 'Age_bin' in X_train.columns else []) if c in X_train.columns])
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# ---------------------------
# 6) IMBALANCE HANDLING
# ---------------------------
# We'll use SMOTE to upsample the minority class in training pipeline
smote = SMOTE(random_state=42)

# ---------------------------
# 7) MODELING: Baseline + Ensemble candidates
# ---------------------------

models = {}
# Logistic baseline
models['lr'] = LogisticRegression(max_iter=1000, class_weight='balanced')

# RandomForest
models['rf'] = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')

# GradientBoosting (sklearn)
models['gb'] = GradientBoostingClassifier(n_estimators=200, random_state=42)

# XGBoost if available
if XGBOOST_AVAILABLE:
    models['xgb'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42, n_jobs=-1)
else:
    print("XGBoost not available in environment. Skipping xgb model. Install 'xgboost' to enable it.")

# CatBoost if available
if CATBOOST_AVAILABLE:
    # identify categorical feature names for CatBoost - it's best to pass column indices
    cat_features_for_catboost = [i for i, c in enumerate((numeric_cols + categorical_cols + (['Age_bin'] if 'Age_bin' in X_train.columns else []))) if c in categorical_cols + (['Age_bin'] if 'Age_bin' in X_train.columns else [])]
    models['catboost'] = CatBoostClassifier(iterations=500, verbose=0, random_state=42)
else:
    print("CatBoost not available. Install 'catboost' if you want to run it.")

# ---------------------------
# 8) TRAIN & EVALUATE FUNCTION
# ---------------------------
def train_and_evaluate(name, estimator, X_train, y_train, X_test, y_test, preprocessor, use_smote=True):
    print(f"\nTraining model: {name}")
    
    # pipeline: preprocessor -> smote (fit only on training) -> estimator
    if use_smote:
        pipe = ImbPipeline(steps=[
            ('pre', preprocessor),
            ('smote', smote),
            ('clf', estimator)
        ])
    else:
        pipe = Pipeline(steps=[
            ('pre', preprocessor),
            ('clf', estimator)
        ])
    # fit
    pipe.fit(X_train, y_train)
    
    # predict probabilities
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"{name} AUC: {auc:.4f}  AP: {ap:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Save pipeline
    joblib.dump(pipe, f"model_{name}.joblib")
    print(f"Saved pipeline to model_{name}.joblib")
    
    # Optional: calibration / PR curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.savefig(f"roc_{name}.png", bbox_inches='tight')
    plt.close()
    
    return {'name': name, 'auc': auc, 'ap': ap, 'pipeline': pipe}

# ---------------------------
# 9) RUN MODELS (loop)
# ---------------------------
results = []
for name, estimator in models.items():
    try:
        res = train_and_evaluate(name, estimator, X_train, y_train, X_test, y_test, preprocessor, use_smote=True)
        results.append(res)
    except Exception as e:
        print(f"Error training {name}: {e}")

# Summarize
summary_df = pd.DataFrame([{'model': r['name'], 'auc': r['auc'], 'ap': r['ap']} for r in results])
print("\nModel performance summary:\n", summary_df.sort_values('auc', ascending=False))

# Save summary
summary_df.to_csv("model_performance_summary.csv", index=False)

# ---------------------------
# 10) FEATURE IMPORTANCE & SHAP (if available)
# ---------------------------
best = max(results, key=lambda x: x['auc'])
print("Best model:", best['name'], "AUC:", best['auc'])

# If the model has feature_importances_ (tree models) we can extract them (but need preprocessor transformed feature names)
def get_feature_names(preprocessor):
    # approximate feature names after ColumnTransformer
    num_features = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    # get onehot names
    try:
        onehot = cat_transformer.named_steps['onehot']
        cat_onehot_names = onehot.get_feature_names_out(cat_features).tolist()
    except Exception:
        cat_onehot_names = cat_features
    return list(num_features) + cat_onehot_names

try:
    feature_names = get_feature_names(preprocessor)
except Exception:
    feature_names = None

clf_pipe = best['pipeline']
try:
    model = clf_pipe.named_steps['clf']
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        fi = model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
        fi_df = fi_df.sort_values('importance', ascending=False).head(30)
        fi_df.to_csv("feature_importance.csv", index=False)
        print("Saved top feature importances to feature_importance.csv")
    else:
        print("Model does not expose feature_importances_ or feature names not available.")
except Exception as e:
    print("Could not extract feature importances:", e)

if SHAP_AVAILABLE:
    print("Generating SHAP summary for best model...")
    explainer = shap.Explainer(clf_pipe.named_steps['clf'], clf_pipe.named_steps['pre'].transform(X_train))
    shap_values = explainer(clf_pipe.named_steps['pre'].transform(X_test))
    shap.summary_plot(shap_values, features=clf_pipe.named_steps['pre'].transform(X_test), feature_names=feature_names, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()
    print("Saved SHAP summary plot shap_summary.png")
else:
    print("SHAP not installed. Skipping SHAP explainability. Install 'shap' to enable.")

# ---------------------------
# 11) LIFT CHART & DECILE ANALYSIS
# ---------------------------
# Create deciles by predicted proba in test and compute lift
best_pipe = best['pipeline']
y_proba_test = best_pipe.predict_proba(X_test)[:,1]
deciles = pd.qcut(y_proba_test, 10, labels=False, duplicates='drop')
decile_df = pd.DataFrame({'y_true': y_test.values, 'y_proba': y_proba_test, 'decile': deciles})
lift = decile_df.groupby('decile').agg({'y_true': ['sum','count'], 'y_proba': 'mean'})
lift.columns = ['defaults','count','avg_proba']
lift = lift.sort_index(ascending=False).reset_index()
lift['default_rate'] = lift['defaults'] / lift['count']
baseline = y_test.mean()
lift['lift'] = lift['default_rate'] / baseline
lift.to_csv("decile_lift.csv", index=False)
print("Saved decile lift to decile_lift.csv")

# ---------------------------
# 12) EXPORT FOR TABLEAU / REPORTING
# ---------------------------
# Save test set with predictions and important columns for dashboarding
X_test_out = X_test.copy()
X_test_out['y_true'] = y_test.values
X_test_out['y_proba'] = y_proba_test
X_test_out.to_csv("test_with_predictions.csv", index=False)
print("Saved test_with_predictions.csv - ready for Tableau import")

# ---------------------------
# 13) FINAL NOTES & NEXT STEPS
# ---------------------------
print("""
Pipeline complete.

Next recommended steps:
- Hyperparameter tuning with RandomizedSearchCV or Optuna for XGBoost/CatBoost/RandomForest.
- Use target encoding (e.g., category_encoders' TargetEncoder) for high-cardinality variables instead of one-hot.
- Build a calibration model (Platt scaling / isotonic) if predicted probabilities need calibration.
- Implement model monitoring (population drift, score distribution) in production.
- If required, implement ROSE via R or rpy2.
- If deploying, package the best pipeline into a Docker image and expose a prediction API (FastAPI).
""")
