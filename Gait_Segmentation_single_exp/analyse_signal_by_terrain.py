import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE = {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+11.6, "distribution": "Stack Front", "control_mode": "NPA", "position": "front"},

# terrain segments distance covered (start, end)
flat = [(0, 1.73), (5.79, 8.97), (13.00, 17.33), (21.53, 24.29), (28.44, 33.25), (37.18, 40.50), (44.13, 100000)]
ascending = []
descending = []

