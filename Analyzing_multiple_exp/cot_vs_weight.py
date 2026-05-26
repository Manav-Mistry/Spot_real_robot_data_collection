from cost_of_transport import compute_power, find_distance_covered, find_ave_velocity, BASELINE_DISTANCE, DISTANCE_PER_LOOP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        total_distance = file["loop"] * DISTANCE_PER_LOOP
        total_energy = compute_power(df)
        avg_velocity = find_ave_velocity(df, file["loop"], total_distance)

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
        total_distance = file["loop"] * DISTANCE_PER_LOOP
        total_energy = compute_power(df)
        avg_velocity = find_ave_velocity(df, file["loop"], total_distance)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)

        print(f"[{file['exp_name']}]  (Energy): {total_energy:.4f} ")

        results_PA.append({
            "name": file["exp_name"],
            "mech_energy": total_energy,
            "cot": round(cot_energy, 3),
            "payload": file["mass"],
            "velocity": avg_velocity,
            "distance": total_distance
        })

    # --- plot CoT vs payload (NPA vs PA)
    payload_weight_NPA = [result["payload"]-33.8-(0 if result["name"] == "baseline" else 5.6) for result in results_NPA]
    payload_weight_PA  = [result["payload"]-33.8-(0 if result["name"] == "baseline" else 5.6) for result in results_PA]

    CoT_PA  = [result["cot"] for result in results_PA]
    CoT_NPA = [result["cot"] for result in results_NPA]

    energy_NPA = [result["mech_energy"] for result in results_NPA]
    energy_PA = [result["mech_energy"] for result in results_PA]
    
    fig, axe = plt.subplots(2, 1, figsize=(10, 6))
    axe[0].plot(payload_weight_NPA, CoT_NPA, "-o",color="black",label="Not Payload Aware")
    axe[0].plot(payload_weight_PA,  CoT_PA, "--o",color="black", label="Payload Aware")
    axe[0].set_ylabel("Cost of Transport")

    axe[1].plot(payload_weight_NPA, energy_NPA, "-o", color="black", label="Not Payload Aware")
    axe[1].plot(payload_weight_PA, energy_PA, "--o", color="black", label="Payload Aware")
    axe[1].set_ylabel("Mechanical Energy (J)")

    fig.legend(*axe[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.98, 0.90), borderpad=0.8)

    axe[0].grid(True)
    axe[1].grid(True)
    # axe[2].grid(True)

    axe[0].set_xticks(payload_weight_PA)
    axe[1].set_xticks(payload_weight_PA)
    # axe[2].set_xticks(payload_weight_PA)
    axe[0].tick_params(labelbottom=False)
    axe[1].tick_params(labelbottom=False)

    fig.supxlabel("Payload Weight (kg)")

    # fig.suptitle("Adjacent Payload Distribution, Incline Flat")
    fig.text(0.1, 0.90, r"$CoT = \frac{Mechanical\ Energy}{Weight \times Distance\ Covered}$",
             ha="left", va="top", fontsize=14)
    plt.tight_layout()
    plt.savefig("cot_vs_weight.png", dpi=300, bbox_inches='tight')
    plt.show()

    

    

