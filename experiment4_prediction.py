"""
=====================================================================
EXPERIMENT 4: Breast Cancer Subtype Prediction Model
=====================================================================
Builds SVM and Logistic Regression models for:
  - Mammaprint_type prediction (NKI70_Good vs NKI70_Bad)
  - GHI_RS_Score prediction (High vs Low, binarized at median)
Both with 5-fold cross-validation.

Dataset: dataset.xlsx (72 samples, 177 columns)
=====================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\Asus\PycharmProjects\medical"
DOWNLOADS = r"C:\Users\Asus\Downloads"
EXP3_RESULTS = os.path.join(BASE_DIR, "experiment3_results")
EXP4_RESULTS = os.path.join(BASE_DIR, "experiment4_results")

os.makedirs(EXP4_RESULTS, exist_ok=True)


def load_clinical_data():
    """
    Search all spreadsheet files in Downloads for Mammaprint_type
    and GHI_RS_Score columns.
    """
    print("  Searching Downloads folder for clinical data...")

    spreadsheets = []
    for f in os.listdir(DOWNLOADS):
        if f.endswith(('.xlsx', '.xls', '.csv')):
            spreadsheets.append(os.path.join(DOWNLOADS, f))
            print(f"    Found: {f}")

    for filepath in spreadsheets:
        try:
            fname = os.path.basename(filepath)
            if fname.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            elif fname.endswith('.xls'):
                df = pd.read_excel(filepath, engine='xlrd')
            else:
                df = pd.read_csv(filepath)

            cols_lower = [c.lower() for c in df.columns]
            has_mamma = any('mammaprint' in c for c in cols_lower)
            has_ghi = any('ghi' in c for c in cols_lower)

            if has_mamma or has_ghi:
                print(f"\n  FOUND TARGET COLUMNS in: {fname}")
                print(f"    Shape: {df.shape}")
                return df, fname

        except Exception as e:
            print(f"    Error reading {os.path.basename(filepath)}: {e}")

    return None, None


def find_target_column(df, keywords):
    """Find a column matching any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def prepare_data(df, target_col):
    """
    Prepare feature matrix X and target vector y.
    Handles both string and numeric target columns correctly.
    """
    y_raw = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y_raw.notna()
    df_valid = df[valid].copy()
    y_raw = y_raw[valid].copy()

    # Encode target
    # Use is_numeric_dtype to correctly handle all string types
    # including newer Pandas StringDtype
    le = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        # Categorical/string target (e.g. NKI70_Good, NKI70_Bad)
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        print(f"    Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        # Numeric target - binarize at median for classification
        y_numeric = pd.to_numeric(y_raw, errors='coerce')
        median = float(y_numeric.median())
        y = (y_numeric > median).astype(int).values
        print(f"    Binarized at median ({median:.2f}): 0=Low, 1=High")

    # Select feature columns (numeric, non-target, non-ID)
    skip_patterns = ['id', 'patient', 'name', 'date', 'unnamed', 'index',
                     'bcr', 'uuid', 'barcode']
    skip_patterns.append(target_col.lower())

    # Also skip the other target column to avoid data leakage
    skip_patterns.append('mammaprint')
    skip_patterns.append('ghi')

    feature_cols = []
    for col in df_valid.columns:
        cl = col.lower()

        # Skip non-feature columns
        if any(s in cl for s in skip_patterns):
            continue

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df_valid[col]):
            if df_valid[col].nunique() > 1:
                feature_cols.append(col)
        else:
            # Try converting to numeric
            converted = pd.to_numeric(df_valid[col], errors='coerce')
            if converted.notna().sum() > len(df_valid) * 0.5 and converted.nunique() > 1:
                df_valid[col] = converted
                feature_cols.append(col)

    X = df_valid[feature_cols].copy()

    # Fill missing values with column median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

    # Drop columns that are still all NaN
    X = X.dropna(axis=1, how='all')

    # Final conversion to float
    X = X.astype(float)

    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"    Sample features: {list(X.columns[:5])}")

    return X.values, y, list(X.columns), le


def build_model(X, y, model_name, target_name):
    """Build and evaluate a model with 5-fold stratified CV."""
    print(f"\n  {model_name} for {target_name}...")

    n_classes = len(np.unique(y))
    min_class = min(np.bincount(y))

    # Use fewer folds if not enough samples per class
    n_folds = min(5, min_class)
    if n_folds < 2:
        print(f"    WARNING: Too few samples in smallest class ({min_class}).")
        print(f"    Skipping this target.")
        return None

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    n_features = min(20, X.shape[1])

    if model_name == 'SVM':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=n_features)),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        probability=True, random_state=42))
        ])
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=n_features)),
            ('clf', LogisticRegression(max_iter=5000, C=1.0,
                                       random_state=42, solver='lbfgs'))
        ])

    # Cross-validation
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    try:
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')
    except:
        y_prob = None

    if n_classes == 2:
        try:
            auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        except:
            auc_scores = None
    else:
        auc_scores = None

    # Print results
    print(f"    {n_folds}-Fold CV Accuracy: {acc_scores}")
    print(f"    Mean Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
    if auc_scores is not None:
        print(f"    Mean AUC:      {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

    print(f"\n    Classification Report:")
    print(classification_report(y, y_pred, digits=4))

    return {
        'model': model_name,
        'target': target_name,
        'acc_scores': acc_scores,
        'mean_acc': acc_scores.mean(),
        'std_acc': acc_scores.std(),
        'auc_scores': auc_scores,
        'mean_auc': auc_scores.mean() if auc_scores is not None else None,
        'y_true': y,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': confusion_matrix(y, y_pred),
        'n_classes': n_classes,
        'n_folds': n_folds,
    }


def create_visualizations(results, save_dir):
    """Generate all charts for the report."""

    # Group results by target
    targets = {}
    for r in results:
        t = r['target']
        if t not in targets:
            targets[t] = []
        targets[t].append(r)

    # ---- Chart 1: Accuracy comparison ----
    n_targets = len(targets)
    fig, axes = plt.subplots(1, max(n_targets, 1), figsize=(7 * n_targets, 6))
    if n_targets == 1:
        axes = [axes]

    for idx, (target, res_list) in enumerate(targets.items()):
        ax = axes[idx]
        names = [r['model'] for r in res_list]
        means = [r['mean_acc'] for r in res_list]
        stds = [r['std_acc'] for r in res_list]
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=colors[:len(names)], edgecolor='black', linewidth=0.5)
        ax.set_title(f'{target}\n{res_list[0]["n_folds"]}-Fold CV Accuracy',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')

        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + 0.02,
                    f'{m:.3f}', ha='center', fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"), dpi=150)
    plt.show()
    print(f"  Saved: accuracy_comparison.png")

    # ---- Chart 2: Confusion matrices ----
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, r in enumerate(results):
        ax = axes[idx]
        cm = r['cm']
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{r['model']}\n({r['target']})", fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=150)
    plt.show()
    print(f"  Saved: confusion_matrices.png")

    # ---- Chart 3: ROC curves ----
    binary = [r for r in results if r['n_classes'] == 2 and r['y_prob'] is not None]
    if binary:
        fig, axes = plt.subplots(1, max(len(targets), 1), figsize=(7 * len(targets), 6))
        if len(targets) == 1:
            axes = [axes]

        for idx, (target, res_list) in enumerate(targets.items()):
            ax = axes[idx]
            colors = ['#3498db', '#e74c3c']
            for i, r in enumerate(res_list):
                if r['y_prob'] is not None and r['n_classes'] == 2:
                    fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'][:, 1])
                    auc = r['mean_auc'] if r['mean_auc'] else 0
                    ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                            label=f"{r['model']} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
            ax.set_title(f'ROC Curve - {target}', fontsize=13, fontweight='bold')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right')
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
        plt.show()
        print(f"  Saved: roc_curves.png")

    # ---- Chart 4: Per-fold results ----
    fig, ax = plt.subplots(figsize=(12, 6))
    max_folds = max(r['n_folds'] for r in results)
    x = np.arange(max_folds)
    width = 0.18
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, r in enumerate(results):
        offset = (i - len(results) / 2 + 0.5) * width
        label = f"{r['model']} ({r['target'][:12]})"
        ax.bar(x[:r['n_folds']] + offset, r['acc_scores'], width,
               label=label, color=colors[i % len(colors)],
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation Results per Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i + 1}' for i in range(max_folds)])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_folds_detail.png"), dpi=150)
    plt.show()
    print(f"  Saved: cv_folds_detail.png")


def save_report(results, save_dir):
    """Save text report and CSV summary."""
    report_path = os.path.join(save_dir, "prediction_results_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  EXPERIMENT 4: BREAST CANCER SUBTYPE PREDICTION RESULTS\n")
        f.write("  Dataset: dataset.xlsx (72 patients, 177 features)\n")
        f.write("=" * 80 + "\n")

        for r in results:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  Model: {r['model']}\n")
            f.write(f"  Target: {r['target']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"  {r['n_folds']}-Fold Cross-Validation Accuracy:\n")
            for i, a in enumerate(r['acc_scores']):
                f.write(f"    Fold {i + 1}: {a:.4f}\n")
            f.write(f"    Mean:   {r['mean_acc']:.4f} +/- {r['std_acc']:.4f}\n")
            if r['mean_auc'] is not None:
                f.write(f"\n  Mean AUC: {r['mean_auc']:.4f}\n")
                if r['auc_scores'] is not None:
                    for i, a in enumerate(r['auc_scores']):
                        f.write(f"    Fold {i + 1}: {a:.4f}\n")
            f.write(f"\n  Confusion Matrix:\n{r['cm']}\n")
            f.write(f"\n  Classification Report:\n")
            f.write(classification_report(r['y_true'], r['y_pred'], digits=4))
            f.write("\n")

    print(f"  Report saved: {report_path}")

    csv_path = os.path.join(save_dir, "prediction_results.csv")
    rows = []
    for r in results:
        row = {
            'Model': r['model'],
            'Target': r['target'],
            'Mean_Accuracy': r['mean_acc'],
            'Std_Accuracy': r['std_acc'],
        }
        if r['mean_auc'] is not None:
            row['Mean_AUC'] = r['mean_auc']
        for i, a in enumerate(r['acc_scores']):
            row[f'Fold{i + 1}_Accuracy'] = a
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  EXPERIMENT 4: Breast Cancer Subtype Prediction")
    print("=" * 70)

    # Step 1: Load clinical data
    print("\n[Step 1] Loading clinical data...")
    df, filename = load_clinical_data()

    if df is None:
        print("  ERROR: No spreadsheet files found in Downloads!")
        sys.exit(1)

    print(f"\n  Using: {filename}")
    print(f"  Shape: {df.shape}")

    # Step 2: Find target columns
    print("\n[Step 2] Finding target columns...")

    mamma_col = find_target_column(df, ['mammaprint'])
    ghi_col = find_target_column(df, ['ghi_rs', 'ghi_score', 'ghirs', 'ghi'])

    if mamma_col:
        print(f"  Mammaprint column: '{mamma_col}'")
        counts = df[mamma_col].dropna().astype(str).value_counts()
        print(f"    Values: {counts.to_dict()}")
    else:
        print("  Mammaprint column: NOT FOUND")

    if ghi_col:
        print(f"  GHI_RS_Score column: '{ghi_col}'")
        vals = pd.to_numeric(df[ghi_col], errors='coerce').dropna()
        if len(vals) > 0:
            print(f"    Range: [{vals.min():.2f}, {vals.max():.2f}], Median: {vals.median():.2f}")
        else:
            print(f"    Values: {df[ghi_col].dropna().astype(str).value_counts().to_dict()}")
    else:
        print("  GHI_RS_Score column: NOT FOUND")

    if mamma_col is None and ghi_col is None:
        print("\n  ERROR: Neither target column found!")
        print("  Available columns:")
        for c in df.columns:
            print(f"    - {c}")
        sys.exit(1)

    # Step 3: Build models
    print("\n[Step 3] Building prediction models...")
    all_results = []

    targets = {}
    if mamma_col:
        targets['Mammaprint_type'] = mamma_col
    if ghi_col:
        targets['GHI_RS_Score'] = ghi_col

    for target_name, col_name in targets.items():
        print(f"\n{'=' * 50}")
        print(f"  TARGET: {target_name}")
        print(f"{'=' * 50}")

        X, y, feat_names, le = prepare_data(df, col_name)

        if X.shape[0] < 10:
            print(f"  Skipping: too few samples ({X.shape[0]})")
            continue

        # SVM
        r1 = build_model(X, y, 'SVM', target_name)
        if r1:
            all_results.append(r1)

        # Logistic Regression
        r2 = build_model(X, y, 'Logistic Regression', target_name)
        if r2:
            all_results.append(r2)

    if not all_results:
        print("\n  ERROR: No models were built!")
        sys.exit(1)

    # Step 4: Visualize
    print("\n[Step 4] Creating visualizations...")
    create_visualizations(all_results, EXP4_RESULTS)

    # Step 5: Save report
    print("\n[Step 5] Saving report...")
    save_report(all_results, EXP4_RESULTS)

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\n  {'Model':<25} {'Target':<20} {'Accuracy':<18} {'AUC'}")
    print("  " + "-" * 75)
    for r in all_results:
        auc = f"{r['mean_auc']:.4f}" if r['mean_auc'] is not None else "N/A"
        print(f"  {r['model']:<25} {r['target']:<20} "
              f"{r['mean_acc']:.4f} +/- {r['std_acc']:.4f}  {auc}")

    print(f"\n  Results in: {EXP4_RESULTS}")
    for f in sorted(os.listdir(EXP4_RESULTS)):
        print(f"    - {f}")
    print("=" * 70)