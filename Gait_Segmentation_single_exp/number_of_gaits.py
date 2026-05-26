import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gait_segmentation import segment_gait_cycles

FILES_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_NPA/incline_flat_12kg_adj_center_NPA_joints_20260426_155231.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_NPA/incline_flat_14kg_adj_center_NPA_joints_20260426_171621.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3}, 
]

FILES_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adjacent_center_16kg", "mass": 33.8 + 16. + 5.6, "loop": 3}, 
]

if __name__ == "__main__":
    results_npa = {}
    results_pa = {}

    for file in FILES_NPA:
        df = pd.read_csv(file["data_file"])
        segments = segment_gait_cycles(df)
        results_npa[file["exp_name"]] = len(segments)
    
    for file in FILES_PA:
        df = pd.read_csv(file["data_file"])
        segments = segment_gait_cycles(df)
        results_pa[file["exp_name"]] = len(segments)

    print("---------- NPA --------------")
    for name, result in results_npa.items():
        print(f"{name} - {result}")
    
    print("---------- PA --------------")
    for name, result in results_pa.items():
        print(f"{name} - {result}")