from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import os

dataset_name = 'ml1m'
scores_folder_name = 'scores'
# Create Result folder structure
if not os.path.exists(scores_folder_name):
    os.mkdir(scores_folder_name)

prep_scores = pd.read_csv(os.path.join(scores_folder_name, '%s_prep.csv' % dataset_name), delimiter='|', index_col=None)
repro_scores = pd.read_csv(os.path.join(scores_folder_name, '%s_repro.csv' % dataset_name), delimiter='|', index_col=None)

for col in repro_scores.columns:
    prep_data = prep_scores[col]
    repro_data = repro_scores[col]
    mean_val = np.mean(repro_data)
    std_val = np.std(repro_data)

    p_r = ttest_ind(prep_data, repro_data)
    print('%s: mean: %1.4f, std: %1.4f statistic: %1.4f, pvalue: %1.4f' % (col, mean_val, std_val, p_r[0], p_r[1]))
