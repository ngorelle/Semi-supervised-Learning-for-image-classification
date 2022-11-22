import numpy as np
import pandas as pd
import os

from main import RESULTS_DIR


def join(*paths):
    return os.path.join(*paths)

csv_files = np.array([join(RESULTS_DIR, filename) for filename in os.listdir(RESULTS_DIR)])
mask = np.array(['_supervised' in filename for filename in csv_files])
dfs = np.array([pd.read_csv(filename) for filename in csv_files])
dfs_supervised = dfs[mask]
csv_files_supervised = csv_files[mask]

dfs_unsupervised = dfs[~mask]
csv_files_unsupervised = csv_files[~mask]
supervised_mask = np.array([df.student_acc_test.max() > .40 for df in dfs_supervised])
unsupervised_mask = np.array([df.teacher_acc_test.max() > .50 for df in dfs_unsupervised])

dfs_supervised = dfs_supervised[supervised_mask]
dfs_unsupervised = dfs_unsupervised[unsupervised_mask]
csv_files_supervised = csv_files_supervised[supervised_mask]
csv_files_unsupervised = csv_files_unsupervised[unsupervised_mask]

with open("supervised.txt", "w") as file:
    file.write('\n'.join(csv_files_supervised.tolist()))

with open("unsupervised.txt", "w") as file:
    file.write('\n'.join(csv_files_unsupervised.tolist()))
