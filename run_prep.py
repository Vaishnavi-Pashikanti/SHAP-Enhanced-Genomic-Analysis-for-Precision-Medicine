import pandas as pd
from data_prep import load_data, prepare_train_test, save_artifacts

# 1. Load your data (update the path if needed)
df = load_data('data/brca_metabric_clinical_data.csv')

# 2. Prepare train/test splits and preprocessing artifacts
X_train, X_test, y_train, y_test, artifacts = prepare_train_test(df)

# 3. Save preprocessing artifacts for later use
save_artifacts(artifacts, out_dir='models')