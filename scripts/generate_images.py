"""
Generate README images: heatmap, confusion matrix, model comparison.
Run from project root: python scripts/generate_images.py
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Project root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, 'images')
DATA_PATH = os.path.join(ROOT, 'data', 'loan_train.csv')
MODEL_PATH = os.path.join(ROOT, 'bin', 'xgboostModel.pkl')

os.makedirs(IMAGES_DIR, exist_ok=True)

def load_and_preprocess():
    """Load train data and preprocess to match notebook (one-hot, drop Loan_ID)."""
    df = pd.read_csv(DATA_PATH)
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    # Drop rows where target is missing
    df = df[df['Loan_Status'].notna()].copy()
    # Fill numeric missing
    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'CoapplicantIncome']:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    # Fill categorical
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) else 'Unknown')
    # Dependents: normalize to 0, 1, 2, 3+
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(str).str.replace('3', '3+')
    cols_obj = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    # Credit_History as object for get_dummies
    df['Credit_History'] = df['Credit_History'].astype(str)
    df = pd.get_dummies(df, columns=cols_obj, drop_first=True)
    return df

def plot_heatmap(df, out_path):
    """Feature correlation heatmap."""
    # Exclude target for correlation
    cols = [c for c in df.columns if c != 'Loan_Status']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved', out_path)

def plot_confusion_matrix(model, X_val, y_val, out_path):
    """Confusion matrix for the given model on validation set."""
    pred = model.predict(X_val)
    cm = confusion_matrix(y_val, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title('Confusion Matrix (XGBoost)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved', out_path)

def plot_model_comparison(df, out_path):
    """Compare XGBoost vs Logistic Regression (accuracy & F1)."""
    feature_cols = [c for c in df.columns if c != 'Loan_Status']
    X = df[feature_cols]
    y = df['Loan_Status'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model 1: XGBoost (if available) or Random Forest
    acc1, f1_1 = 0.0, 0.0
    name1 = 'XGBoost'
    if os.path.isfile(MODEL_PATH):
        try:
            m1 = joblib.load(open(MODEL_PATH, 'rb'))
            pred1 = m1.predict(X_val)
            acc1 = accuracy_score(y_val, pred1)
            f1_1 = f1_score(y_val, pred1, zero_division=0)
        except Exception as e:
            print('Could not load XGBoost for comparison:', e)
            name1 = 'Random Forest'
            m1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            pred1 = m1.predict(X_val)
            acc1 = accuracy_score(y_val, pred1)
            f1_1 = f1_score(y_val, pred1, zero_division=0)
    else:
        name1 = 'Random Forest'
        m1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        pred1 = m1.predict(X_val)
        acc1 = accuracy_score(y_val, pred1)
        f1_1 = f1_score(y_val, pred1, zero_division=0)

    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_pred)
    lr_f1 = f1_score(y_val, lr_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    metrics = ['Accuracy', 'F1 Score']
    vals1 = [acc1, f1_1]
    lr_vals = [lr_acc, lr_f1]
    ax.bar(x - width/2, vals1, width, label=name1)
    ax.bar(x + width/2, lr_vals, width, label='Logistic Regression')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.title('Model Comparison (validation set)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved', out_path)

def main():
    sys.path.insert(0, ROOT)
    df = load_and_preprocess()

    # 1. Heatmap
    plot_heatmap(df, os.path.join(IMAGES_DIR, 'heatmap.png'))

    # 2. Confusion matrix (XGBoost on validation, or LR if model missing/unloadable)
    feature_cols = [c for c in df.columns if c != 'Loan_Status']
    X = df[feature_cols]
    y = df['Loan_Status'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    model = None
    if os.path.isfile(MODEL_PATH):
        try:
            model = joblib.load(open(MODEL_PATH, 'rb'))
        except Exception as e:
            print('Could not load XGBoost model:', e)
    if model is None:
        model = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
        print('Using Logistic Regression for confusion matrix (XGBoost model not loaded).')
    plot_confusion_matrix(model, X_val, y_val, os.path.join(IMAGES_DIR, 'confusion_matrix.png'))

    # 3. Model comparison
    plot_model_comparison(df, os.path.join(IMAGES_DIR, 'model_comparison.png'))

    print('Done. Images saved to', IMAGES_DIR)

if __name__ == '__main__':
    main()
