import os
import sys

import pandas as pd

sys.stdout = open("txt_files/dataset_frequency.txt", "w")

repo_path = "./../../main_dataset/count_data/"

for root, dirs, files in os.walk(repo_path):
    for file in files:
        file_path = os.path.join(root, file)
        df : pd.DataFrame= pd.read_csv(file_path)
        first_row = df.iloc[0]["date"]
        last_row = df.iloc[-1]["date"]
        print(file, first_row, last_row)
