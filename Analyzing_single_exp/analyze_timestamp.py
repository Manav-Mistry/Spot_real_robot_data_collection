import numpy as np
import pandas as pd

DATA_FILE = "/home/nerve/Desktop/data_collected/Mar_10/Test/multiple_autowalk_test_run01_joints_20260310_155322.csv"

df = pd.read_csv(DATA_FILE)

df_time_stamp = df["timestamp"] - df["timestamp"][0]
print(df_time_stamp[333: 350])