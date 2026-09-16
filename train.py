import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score

import joblib

import mlflow
import mlflow.sklearn

from app.eda import run_full_eda, plot_feature_importance_bar
from app.config import FEATURE_DESCRIPTIONS



RAW_PATH = "data\kepler_koi.csv"
TARGET = "koi_disposition"
LEAKAGE_COLS = [
    "koi_pdisposition", "koi_score", "koi_disp_prov", "koi_comment",
    "koi_datalink_dvr", "koi_datalink_dvs", "koi_vet_date", "koi_vet_stat",
]
ID_COLS = ["rowid", "kepid", "kepoi_name", "kepler_name"]

def load_and_clean_raw():
    df = pd.read_csv(RAW_PATH, comment="#")
    bad_mask = ~df["koi_fpflag_nt"].isin([0, 1])
    if bad_mask.sum():
        df.loc[bad_mask, "koi_fpflag_nt"] = 1
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols)
    null_frac = df.isnull().mean()
    mostly_null = null_frac[null_frac > 0.9].index.tolist()
    df = df.drop(columns=[c for c in mostly_null if c in df.columns])
    return df


def get_feature_target(df: pd.DataFrame):
    drop_cols = [c for c in (LEAKAGE_COLS + ID_COLS) if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET])
    y = df[TARGET]
    groups = df["kepid"]  
    return X, y, groups


def group_stratified_split(X, y, groups, test_size=0.2, val_size=0.1, random_state=42):
    group_df = pd.DataFrame({"kepid": groups, "y": y})
    group_majority = group_df.groupby("kepid")["y"].agg(lambda s: s.value_counts().idxmax())
    train_val_groups, test_groups = train_test_split(
        group_majority.index, test_size=test_size,
        stratify=group_majority.values, random_state=random_state
    )
    train_val_majority = group_majority.loc[train_val_groups]
    relative_val_size = val_size / (1 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups, test_size=relative_val_size,
        stratify=train_val_majority.values, random_state=random_state
    )
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    test_mask = groups.isin(test_groups)
    return (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])

FPFLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]


def evaluate(name, y_true, y_pred, labels):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    print(f"Macro-F1: {macro_f1:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\nConfusion matrix (rows=true, cols=pred), labels={labels}:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    return macro_f1


def calc_baseline():
    df = load_and_clean_raw()
    X, y, groups = get_feature_target(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = group_stratified_split(X, y, groups)

    labels = sorted(y.unique())

    # --- Baseline 1: majority class ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_val)
    f1_dummy = evaluate("BASELINE 1: Majority class", y_val, y_pred_dummy, labels)

    # --- Baseline 2: logistic regression on fpflags only ---
    X_train_flags = X_train[FPFLAG_COLS]
    X_val_flags = X_val[FPFLAG_COLS]

    heuristic = LogisticRegression(max_iter=1000)
    heuristic.fit(X_train_flags, y_train)
    y_pred_heuristic = heuristic.predict(X_val_flags)
    f1_heuristic = evaluate(
        "BASELINE 2: Logistic regression on fpflags only",
        y_val, y_pred_heuristic, labels
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Majority-class baseline macro-F1: {f1_dummy:.4f}")
    print(f"FPflag-only heuristic macro-F1:   {f1_heuristic:.4f}")
    print(f"\n=> Any real model must clear {f1_heuristic:.4f} by a meaningful margin")
    print("   to justify its added complexity over 4 boolean columns.")

    joblib.dump(heuristic, "models/baseline_heuristic.joblib")
    with open("models/baseline_scores.txt", "w") as f:
        f.write(f"majority_class_macro_f1={f1_dummy:.4f}\n")
        f.write(f"fpflag_heuristic_macro_f1={f1_heuristic:.4f}\n")
    print("\nSaved baseline model and scores to models/")


BASELINE_MACRO_F1 = 0.5527


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols)], remainder="drop")
    return preprocessor, numeric_cols



def log_confusion_matrix(y_true, y_pred, labels, run_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {run_name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    path = f"models/confusion_matrix_{run_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_experiment(name, model, X_train, y_train, X_val, y_val, preprocessor, numeric_cols, params):
    labels = sorted(y_train.unique())
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_features", len(numeric_cols))
 
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
 
        macro_f1 = f1_score(y_val, y_pred, labels=labels, average="macro", zero_division=0)
        confirmed_recall = recall_score(y_val, y_pred, labels=["CONFIRMED"], average="macro", zero_division=0)
        report = classification_report(y_val, y_pred, labels=labels, zero_division=0)
 
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("confirmed_recall", confirmed_recall)
        mlflow.log_metric("beats_baseline", int(macro_f1 > BASELINE_MACRO_F1))
 
        cm_path = log_confusion_matrix(y_val, y_pred, labels, name)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(
            pipeline, "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
 
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(report)
        print(f"Macro-F1: {macro_f1:.4f}  |  CONFIRMED recall: {confirmed_recall:.4f}")
        print(f"Beats baseline ({BASELINE_MACRO_F1}): {macro_f1 > BASELINE_MACRO_F1}")
 
        return pipeline, macro_f1, confirmed_recall


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/eda", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("exoplanet-koi-classification-1")

    df = load_and_clean_raw()

    # --- Run and save Exploratory Data Analysis (EDA) visual plots ---
    eda_plots = run_full_eda(df, output_dir=Path("models/eda"))

    X, y, groups = get_feature_target(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = group_stratified_split(X, y, groups)

    preprocessor, numeric_cols = build_preprocessor(X_train)

    candidates = [
        ("logistic_regression",
         LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"C": 1.0, "class_weight": "balanced"}),
        ("random_forest",
         RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42),
         {"n_estimators": 300, "max_depth": "None", "class_weight": "balanced"}),
        ("gradient_boosting",
         GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
         {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}),
    ]

    results = []
    for name, model, params in candidates:
        pipeline, macro_f1, confirmed_recall = run_experiment(
            name, model, X_train, y_train, X_val, y_val, preprocessor, numeric_cols, params
        )
        results.append((name, pipeline, macro_f1, confirmed_recall))

    results.sort(key=lambda r: r[2], reverse=True)
    best_name, best_pipeline, best_f1, best_recall = results[0]

    print(f"\n{'=' * 60}\nMODEL COMPARISON SUMMARY\n{'=' * 60}")
    for name, _, f1, recall in results:
        marker = " <- BEST" if name == best_name else ""
        print(f"{name:25s} macro_f1={f1:.4f}  confirmed_recall={recall:.4f}{marker}")
    print(f"\nBaseline (fpflags only): {BASELINE_MACRO_F1:.4f}")
    print(f"Best model improvement:  +{best_f1 - BASELINE_MACRO_F1:.4f}")

    import joblib
    joblib.dump(best_pipeline, "models/best_model.joblib")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    import json
    with open("models/feature_columns.json", "w") as f:
        json.dump(numeric_cols, f)
    with open("models/class_labels.json", "w") as f:
        json.dump(sorted(y_train.unique().tolist()), f)

    # --- Plot & Save Feature Importance of Best Model ---
    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        raw_imp = best_pipeline.named_steps["model"].feature_importances_
        total_imp = sum(raw_imp) or 1.0
        imp_list = []
        for feat, w in zip(numeric_cols, raw_imp):
            desc, cat = FEATURE_DESCRIPTIONS.get(feat, (f"Feature {feat}", "other"))
            imp_list.append((feat, float(w / total_imp), cat, desc))
        imp_list.sort(key=lambda x: x[1], reverse=True)
        feat_plot_path = plot_feature_importance_bar(imp_list, Path("models/eda"), top_n=20)
        # Also copy / save to models for direct API access
        plot_feature_importance_bar(imp_list, Path("models"), top_n=20)
        print(f"[EDA] Saved feature importance plot to {feat_plot_path}")

    print(f"\nSaved best model ({best_name}) to models/best_model.joblib")
    print("Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect all logged runs.")
