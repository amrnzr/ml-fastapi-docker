"""Train a customer-churn classifier on the IBM Telco Customer Churn dataset
and save it to model.joblib.

Usage:
    python train.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_customer_churn.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
RANDOM_STATE = 42
TARGET = "Churn"

# customerID is an identifier, gender is a sensitive attribute: both are not used by the model.
DROPPED = ["customerID", "gender"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is text and is blank for 11 brand-new customers (tenure = 0) -> NaN, imputed in the pipeline
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = (df[TARGET] == "Yes").astype(int)  # 1 = customer churned
    X = df[FEATURES]
    return X, y


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def main() -> None:
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, churn rate = {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_auc = None, -1.0
    for name, clf in candidates.items():
        pipe = build_pipeline(clf).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        res = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_churn": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        results[name] = res
        print(f"\n{name}: {res}")
        print(classification_report(y_test, pred, target_names=["stay", "churn"]))
        if res["roc_auc"] > best_auc:
            best_name, best_auc = name, res["roc_auc"]

    # Refit the selected model on all data before saving it as the deployment artifact.
    final_model = build_pipeline(candidates[best_name]).fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    metrics = {
        "dataset": "IBM Telco Customer Churn (7043 customers)",
        "selected_model": best_name,
        "selection_metric": "roc_auc on stratified 20% hold-out",
        "test_metrics": results,
        "features": FEATURES,
        "dropped_columns": DROPPED,
        "classes": {"0": "stay", "1": "churn"},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSelected model: {best_name} (ROC-AUC = {best_auc:.4f})")
    print(f"Saved model to {MODEL_PATH.name} and metrics to {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
