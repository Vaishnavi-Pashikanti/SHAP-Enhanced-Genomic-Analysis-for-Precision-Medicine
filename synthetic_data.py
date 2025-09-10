# synthetic_data.py
import numpy as np
import pandas as pd

def synth_gen(n=1000, n_genes=200, seed=42):
    np.random.seed(seed)
    # gene-like numeric features
    X = np.random.randn(n, n_genes)
    gene_cols = [f"GENE_{i:04d}" for i in range(n_genes)]
    # clinical features
    age = np.random.randint(30, 85, size=n)
    tumor_size = np.random.randint(5, 60, size=n)
    tmb = np.abs(np.random.randn(n)) * 3
    # simple label: linear combo + noise
    linear = (X[:, :10].sum(axis=1) * 0.2 + age*0.03 + tumor_size*0.05 + tmb*0.1)
    prob = 1 / (1 + np.exp(-linear))
    y = (prob > np.median(prob)).astype(int)
    df = pd.DataFrame(X, columns=gene_cols)
    df['Age at Diagnosis'] = age
    df['Tumor Size'] = tumor_size
    df['TMB (nonsynonymous)'] = tmb
    df['Overall Survival Status'] = y
    return df

if __name__ == "__main__":
    df = synth_gen(500, 100)
    df.to_csv("data/synthetic_genomic.csv", index=False)
    print("Saved synthetic to data/synthetic_genomic.csv")
