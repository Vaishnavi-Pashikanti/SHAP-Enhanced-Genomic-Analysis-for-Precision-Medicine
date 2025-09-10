import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib


# -------------------
# LOAD DATA
# -------------------
def load_data(csv_path):
    """Load genomic/clinical data from CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[load_data] Loaded shape={df.shape} from {csv_path}")
    return df


# -------------------
# TARGET MAPPING
# -------------------
def map_survival_status(val):
    """
    Map survival status to binary label:
    0 = living, 1 = deceased
    """
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    if 'deceased' in s or 'died' in s or s.startswith('1'):
        return 1
    if 'living' in s or 'alive' in s or s.startswith('0'):
        return 0
    try:
        v = float(val)
        return 1 if v == 1 else 0
    except Exception:
        return np.nan


# -------------------
# FEATURE SELECTION
# -------------------
def choose_features(df, drop_columns=None, keep_columns=None):
    """Drop leakage & ID columns, keep clinically relevant features."""
    if drop_columns is None:
        drop_columns = [
            # Identifiers
            'Study ID', 'Patient ID', 'Sample ID', 'Sample Type',
            'Oncotree Code',

            # Explicit survival-related (leakage!)
            'Overall Survival (Months)',
            'Overall Survival Status',
            "Patient's Vital Status",
            'Overall Survival Status_bin'
        ]

    # Ensure only valid cols dropped
    drop_columns = [c for c in drop_columns if c in df.columns]

    if keep_columns is not None:
        keep_columns = [c for c in keep_columns if c in df.columns]
        return df[keep_columns].copy()

    return df.drop(columns=drop_columns, errors='ignore').copy()


def build_preprocessor(X, max_categories_onehot=50):
    """Build preprocessing pipelines for numeric & categorical data."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = [c for c in categorical_cols if X[c].nunique() <= max_categories_onehot]
    high_card = [c for c in categorical_cols if X[c].nunique() > max_categories_onehot]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if low_card:
        transformers.append(('cat', categorical_pipeline, low_card))

    if high_card:
        print(f"[build_preprocessor] Dropping high-cardinality categorical cols: {high_card}")

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, numeric_cols, low_card


def get_feature_names_from_column_transformer(ct, numeric_cols, cat_cols):
    """Get feature names after preprocessing (requires fitted ct)."""
    feature_names = []
    for name, transformer, columns in ct.transformers_:
        if name == 'remainder':
            continue
        if transformer is None:
            feature_names.extend(columns)
            continue
        if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
            ohe = transformer.named_steps['onehot']
            feature_names.extend(list(ohe.get_feature_names_out(columns)))
        else:
            feature_names.extend(columns)
    return feature_names


# -------------------
# MAIN PREP FUNCTION
# -------------------
def prepare_train_test(
    df,
    target_col='Overall Survival Status',
    test_size=0.2,
    random_state=42,
    k_best=50
):
    """Preprocess dataset and return train/test splits with artifacts."""

    # Clean target
    df[target_col + '_bin'] = df[target_col].apply(map_survival_status)
    df = df.dropna(subset=[target_col + '_bin']).reset_index(drop=True)

    y = df[target_col + '_bin'].astype(int)
    X = choose_features(df)

    # Drop cols with too many missing values
    missing_frac = X.isna().mean()
    to_drop = missing_frac[missing_frac > 0.5].index.tolist()
    if to_drop:
        print(f"[prepare] Dropping cols with >50% missing: {to_drop}")
        X = X.drop(columns=to_drop)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    # Preprocess
    preprocessor, numeric_cols, cat_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    feature_names = get_feature_names_from_column_transformer(preprocessor, numeric_cols, cat_cols)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Feature selection
    k = min(k_best, X_train_trans.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train_trans, y_train)

    sel_mask = selector.get_support()
    selected_feature_names = [feature_names[i] for i, m in enumerate(sel_mask) if m]

    # Reduce arrays
    X_train_sel = selector.transform(X_train_trans)
    X_test_sel = selector.transform(X_test_trans)

    artifacts = {
        'preprocessor': preprocessor,
        'selector': selector,
        'selected_feature_names': selected_feature_names,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols
    }

    print(f"[prepare] Final train shape={X_train_sel.shape}, test shape={X_test_sel.shape}")
    print(f"[prepare] Selected top {len(selected_feature_names)} features.")

    return X_train_sel, X_test_sel, y_train, y_test, artifacts


# -------------------
# SAVE ARTIFACTS
# -------------------
def save_artifacts(artifacts, out_dir='models'):
    """Save preprocessing pipeline and feature selector artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifacts['preprocessor'], os.path.join(out_dir, 'preprocessor.joblib'))
    joblib.dump(artifacts['selector'], os.path.join(out_dir, 'selector.joblib'))
    with open(os.path.join(out_dir, 'selected_feature_names.json'), 'w') as f:
        json.dump(artifacts['selected_feature_names'], f)
    print(f"[save_artifacts] Artifacts saved to {out_dir}")
