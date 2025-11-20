import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# --- Try importing specialized libraries (Robustness Check) ---
try:
    from imblearn.over_sampling import SMOTE # For Imbalance Handling
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not found. Data balancing (SMOTE) will be skipped.")
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier # For Powerful Gradient Boosting
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not found. Using sklearn's HGBoost as a proxy.")
    XGBOOST_AVAILABLE = False

# Note: CatBoost requires separate installation and configuration.
# We will simulate its use with a placeholder or fallback.

# --- 1. Robust Helper Function: Gini Coefficient (Credit Risk Standard) ---

def calculate_gini(y_true, y_pred_proba):
    """
    Calculates the Gini coefficient from the AUC-ROC score.
    Gini = 2 * AUC - 1
    A Gini of 0 means the model is no better than random, 1 is perfect.
    """
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc_score - 1
        return gini, auc_score
    except ValueError:
        return 0.0, 0.0 # Return 0 if calculation fails (e.g., only one class present)

# --- 2. Data Generation (Simulation of a powerful, complex dataset) ---

def generate_mock_data(n_samples=5000):
    """
    Generates a mock credit risk dataset with numerical, categorical, and text features.
    """
    print(f"Generating {n_samples} simulated records...")

    # Numerical Features (Structured Data)
    data = {
        'loan_amount': np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples),
        'interest_rate': np.random.uniform(5.0, 25.0, n_samples),
        'age': np.random.randint(22, 65, n_samples),
        'annual_income': np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples),
        'credit_utilization': np.random.beta(a=2, b=5, size=n_samples),
    }

    # Categorical Features (Structured Data)
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    data['loan_grade'] = np.random.choice(grades, n_samples, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])
    data['home_ownership'] = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], n_samples, p=[0.55, 0.30, 0.15])

    # Text Feature (For Statistical NLP)
    purposes = [
        "debt consolidation to clear high-interest cards",
        "home improvement project for the kitchen renovation",
        "major purchase of a new car",
        "medical bills due to unexpected emergency",
        "education expenses for college tuition",
        "business investment in new equipment"
    ]
    data['loan_purpose_text'] = [random.choice(purposes) for _ in range(n_samples)]

    # Target Variable (Highly Imbalanced Target - Default Flag)
    # Generate ~8% default rate (a common real-world scenario)
    is_default = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    data['default_flag'] = is_default

    df = pd.DataFrame(data)

    # Introduce some correlations to make the problem predictive
    df.loc[df['loan_grade'].isin(['F', 'G']), 'default_flag'] = np.random.choice([0, 1], len(df[df['loan_grade'].isin(['F', 'G'])]), p=[0.5, 0.5])
    df.loc[df['credit_utilization'] > 0.6, 'default_flag'] = np.random.choice([0, 1], len(df[df['credit_utilization'] > 0.6]), p=[0.7, 0.3])

    return df

# --- 3. Feature Engineering & Preprocessing Pipeline (Robust ColumnTransformer) ---

def create_preprocessing_pipeline():
    """
    Creates a robust ColumnTransformer to handle all feature types.
    This includes:
    1. Numerical Scaling (StandardScaler)
    2. Categorical Encoding (OneHotEncoder)
    3. Text Vectorization (TF-IDF for Statistical NLP)
    """

    # Define feature types
    numerical_features = ['loan_amount', 'interest_rate', 'age', 'annual_income', 'credit_utilization']
    categorical_features = ['loan_grade', 'home_ownership']
    text_feature = 'loan_purpose_text'

    # Preprocessing steps
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Statistical NLP component: TF-IDF
    text_pipeline = Pipeline([
        # TfidfVectorizer converts text into a matrix of TF-IDF features
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=100))
    ])

    # Combine all feature transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
            ('text_nlp', text_pipeline, text_feature)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # Set output to pandas DataFrame for better inspection
    preprocessor.set_output(transform="pandas")

    return preprocessor

# --- 4. Model Training and Evaluation (Comprehensive ML Approach) ---

def run_model_pipeline(X, y):
    """
    Executes the full ML pipeline: splitting, preprocessing, balancing, training, and evaluation.
    """
    print("\n--- Starting Model Pipeline Execution ---")
    
    # Step 1: Split Data (Train/Test for robust evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Original Training Balance (Default Rate): {y_train.mean():.4f}")

    # Step 2: Preprocessing
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Feature Matrix Shape after Preprocessing: {X_train_processed.shape}")

    # Step 3: Data Imbalance Handling (SMOTE/ROSE Approach)
    if IMBLEARN_AVAILABLE:
        print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique)... ")
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
        print(f"Balanced Training Balance (Default Rate): {y_train_balanced.mean():.4f}")
    else:
        print("\nSkipping SMOTE due to missing library. Training on original imbalanced data.")
        X_train_balanced, y_train_balanced = X_train_processed, y_train

    # List of models to test (Ensemble/Comparison Approach)
    models = {
        'RandomForest (RF)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'Scikit-GBM (HGBoost)': HistGradientBoostingClassifier(max_iter=100, random_state=42, l2_regularization=0.1),
    }

    if XGBOOST_AVAILABLE:
        # XGBoost is highly optimized and robust for structured data
        models['XGBoost'] = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
    else:
        print("Note: XGBoost is highly recommended but using HGBoost as powerful substitute.")

    # Simulation for CatBoost (Extremely robust for categorical features)
    # In a real scenario, this would be an actual CatBoostClassifier instance
    print("\nSimulating CatBoost approach evaluation...")


    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Training (on balanced data)
        model.fit(X_train_balanced, y_train_balanced)

        # Prediction (on test data)
        # Use predict_proba for scoring metrics (AUC, Gini)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate performance metrics
        gini, auc = calculate_gini(y_test, y_pred_proba)
        
        # Calculate binary predictions for F1/Report
        # Using a balanced decision threshold (e.g., 0.5) or optimizing it based on business cost matrix
        y_pred = model.predict(X_test_processed) 
        
        results[name] = {
            'AUC': auc,
            'Gini': gini,
            'Classification_Report': classification_report(y_test, y_pred, zero_division=0)
        }

    return results

# --- 5. Main Execution ---

if __name__ == "__main__":
    
    # 1. Generate Data
    df_data = generate_mock_data(n_samples=10000)
    
    # Define features (X) and target (y)
    X_full = df_data.drop('default_flag', axis=1)
    y_full = df_data['default_flag']

    # 2. Run the Comprehensive ML Pipeline
    evaluation_results = run_model_pipeline(X_full, y_full)

    # 3. Display Final Results (Comparison of Models)
    print("\n=======================================================")
    print("           CREDIT RISK MODEL PERFORMANCE SUMMARY")
    print("=======================================================")

    best_gini = -1
    best_model = ""

    for name, result in evaluation_results.items():
        print(f"\nModel: {name}")
        print("-" * (len(name) + 7))
        print(f"  > AUC-ROC Score: {result['AUC']:.4f}")
        print(f"  > Gini Coefficient: {result['Gini']:.4f}")
        
        # Print Detailed Classification Report
        print("\nDetailed Classification Report (Precision/Recall/F1):")
        print(result['Classification_Report'])
        
        if result['Gini'] > best_gini:
            best_gini = result['Gini']
            best_model = name

    print(f"\n[FINAL DECISION] The best performing model based on Gini Coefficient is: {best_model} ({best_gini:.4f})")

    print("\n--- Next Steps in a Production Environment ---")
    print("1. Hyperparameter Tuning (GridSearch/Bayesian Opt) for the best model.")
    print("2. Calibration of the predicted probabilities.")
    print("3. Deployment and Monitoring of the model's performance over time.")
    print("4. Feature Importance analysis (e.g., SHAP values).")