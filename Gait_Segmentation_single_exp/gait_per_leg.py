import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE = "/home/nerve/Desktop/data_collected/incline_crossing_May_13/stack_front_8kg_NPA/incline_crossing_stack_front_8kg_NPA_joints_20260513_170453.csv"

# 1 - ground, 2 - air
def plot_gaits(contact_states):
    fl = contact_states["fl"]
    fr = contact_states["fr"]

    hl = contact_states["hl"]
    hr = contact_states["hr"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 7))

    axes[0].plot(fl)
    axes[0].set_title("fl")

    axes[1].plot(fr)
    axes[1].set_title("fr")

    axes[2].plot(hl)
    axes[2].set_title("hl")

    axes[3].plot(hr)
    axes[3].set_title("hr")

    plt.show()
    return



def extract_contact_states(df):
    fl_foot_contact = df["contact_fl"][5000:10000]
    fr_foot_contact = df["contact_fr"][5000:10000]

    hl_foot_contact = df["contact_hl"][5000:10000]
    hr_foot_contact = df["contact_hr"][5000:10000]

    return {"fl": fl_foot_contact, "fr": fr_foot_contact, "hl": hl_foot_contact, "hr" : hr_foot_contact}

if __name__ == "__main__":

    df = pd.read_csv(FILE)
    contact_states = extract_contact_states(df)
    plot_gaits(contact_states)
    

