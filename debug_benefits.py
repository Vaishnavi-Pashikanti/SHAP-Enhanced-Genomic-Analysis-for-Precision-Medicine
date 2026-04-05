import pandas as pd
from treatment_benefit_estimator import TreatmentBenefitEstimator
from data_prep import choose_features

est = TreatmentBenefitEstimator()
df = pd.read_csv('CheckSample.csv', low_memory=False)
print('loaded', df.shape)
X = choose_features(df)
print('X shape', X.shape)
for i in range(min(3, len(X))):
    row = X.iloc[i]
    benefits, baseline = est.estimate_treatment_benefits(row)
    print('patient', i, 'baseline', baseline.survival_probability, baseline.predicted_survival_months)
    nonzero = False
    for b in benefits:
        if abs(b.survival_probability_benefit) > 1e-6 or abs(b.survival_months_benefit) > 1e-6:
            print('non-zero', b.scenario.to_dict(), b.survival_probability_benefit, b.survival_months_benefit)
            nonzero = True
            break
    if not nonzero:
        print('all zero')
