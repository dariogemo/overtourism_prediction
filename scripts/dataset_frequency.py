import os
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

with open("txt_files/dataset_frequency.txt", "w", encoding="utf-8") as txt_file:
    with redirect_stdout(txt_file), redirect_stderr(txt_file):
        repo_path: str = "./../../main_dataset/count_data/"

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                df: pd.DataFrame = pd.read_csv(file_path)
                first_row: str = df.iloc[0]["date"]
                last_row: str = df.iloc[-1]["date"]
                print(file, first_row, last_row)
