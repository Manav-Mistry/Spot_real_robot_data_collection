from cost_of_transport import compute_power, find_distance_covered, find_ave_velocity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILES_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_NPA/incline_flat_12kg_adj_center_NPA_joints_20260426_155231.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_NPA/incline_flat_14kg_adj_center_NPA_joints_20260426_171621.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14.}, 
]

FILES_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adjacent_center_16kg", "mass": 33.8 + 16.}, 
]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]
FS = 333
DT = 1/FS

TORQUE_COLS = [f"{leg}.{joint}_load" for joint in JOINTS for leg in LEGS]
ANG_VEL_COLS = [f"{leg}.{joint}_velocity" for joint in JOINTS for leg in LEGS]

TORQUE_VELOCITY_PAIR = list(zip(TORQUE_COLS, ANG_VEL_COLS))

if __name__ == "__main__":
    
    results_NPA = []
    # --- find cost of transport for NPA---------------------------
    for file in FILES_NPA:
        df = pd.read_csv(file["data_file"])
        
        if "start" in file:
            start = file["start"]
            end = file["end"]
            df = df.iloc[start: end]
        total_distance= find_distance_covered(df)
        total_energy = compute_power(df)
        avg_velocity = find_ave_velocity(df)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)

        print(f"[{file['exp_name']}] CoT (energy): {cot_energy:.4f} ")

        results_NPA.append({
            "name": file["exp_name"],
            "mech_energy": total_energy,
            "cot": round(cot_energy, 2),
            "payload": file["mass"],
            "velocity": avg_velocity,
            "distance": total_distance
        })


    results_PA = []
    # --- find CoT for PA -----------------------------------------
    for file in FILES_PA:
        df = pd.read_csv(file["data_file"])
        
        if "start" in file:
            start = file["start"]
            end = file["end"]
            df = df.iloc[start: end]
        total_distance= find_distance_covered(df)
        total_energy = compute_power(df)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)

        print(f"[{file['exp_name']}] CoT (energy): {cot_energy:.4f} ")

        results_PA.append({
            "name": file["exp_name"],
            "mech_energy": total_energy,
            "cot": round(cot_energy, 2),
            "payload": file["mass"],
            "velocity": avg_velocity,
            "distance": total_distance
        })

    # --- plot CoT vs payload (NPA vs PA)
    payload_weight_NPA = [result["payload"]-33.8 for result in results_NPA]
    payload_weight_PA  = [result["payload"]-33.8 for result in results_PA]

    CoT_PA  = [result["cot"] for result in results_PA]
    CoT_NPA = [result["cot"] for result in results_NPA]

    vel_NPA = [result["velocity"] for result in results_NPA]
    vel_PA = [result["velocity"] for result in results_PA]

    energy_NPA = [result["mech_energy"] for result in results_NPA]
    energy_PA = [result["mech_energy"] for result in results_PA]
    
    fig, axe = plt.subplots(3, 1)
    axe[0].plot(payload_weight_NPA, CoT_NPA, "-o" ,label="Not Payload Aware")
    axe[0].plot(payload_weight_PA,  CoT_PA, "-o", label="Payload Aware")

    axe[0].set_xlabel("Payload Weight (kg)")
    axe[0].set_ylabel("Cost of Transport")
    axe[0].legend()

    # velocity graph
    axe[1].plot(payload_weight_NPA, vel_NPA, "-o", label="Not Payload Aware")
    axe[1].plot(payload_weight_PA, vel_PA, "-o", label="Payload Aware")
    axe[1].set_ylabel("Average Velocity")
    axe[1].legend()

    # Energy graph

    axe[2].plot(payload_weight_NPA, energy_NPA, "-o", label="Not Payload Aware")
    axe[2].plot(payload_weight_PA, energy_PA, "-o", label="Payload Aware")
    axe[2].set_ylabel("Mechanical Energy")
    axe[2].legend()
    
    axe[0].grid(True)
    axe[1].grid(True)
    axe[2].grid(True)

    axe[0].set_xticks(payload_weight_PA)
    axe[1].set_xticks(payload_weight_PA)
    axe[2].set_xticks(payload_weight_PA)
    
    fig.suptitle("Adjacent Payload Distribution, Incline Flat")
    plt.show()


    # --- distance plot ---
    

    

