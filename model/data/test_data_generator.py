import pandas as pd
import numpy as np
import os

# Configuration
NUM_SAMPLES = 300
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "inference_test_bench.csv"

def generate_unlabeled_test_data():
    np.random.seed(42)

    # ================================
    # BASE HEALTHY DATA
    # ================================
    data = {
        "timestamp": pd.date_range(start="2026-01-01", periods=NUM_SAMPLES, freq="min"),
        "machine_id": ["M_001"] * NUM_SAMPLES,

        "process_temperature": np.random.normal(308, 1, NUM_SAMPLES),
        "air_temperature": np.random.normal(298, 1, NUM_SAMPLES),

        "vibration": np.random.normal(0.2, 0.02, NUM_SAMPLES),
        "torque": np.random.normal(40, 2, NUM_SAMPLES),
        "rpm": np.random.normal(1500, 20, NUM_SAMPLES),

        "current": np.random.normal(5, 0.3, NUM_SAMPLES),

        "operating_hours": np.linspace(100, 400, NUM_SAMPLES),
        "time_since_last_maintenance": np.linspace(10, 200, NUM_SAMPLES),

        "last_maintenance_Type": ["Routine"] * NUM_SAMPLES,

        "idle_duration": np.random.normal(3, 1, NUM_SAMPLES),

        "power_consumption": np.random.normal(2000, 50, NUM_SAMPLES),
    }

    df = pd.DataFrame(data)

    # ================================
    # SUBTLE DEGRADATION
    # ================================
    degrade_range = range(151, 221)

    df.loc[degrade_range, "process_temperature"] += np.linspace(0, 10, len(degrade_range))
    df.loc[degrade_range, "vibration"] += np.linspace(0, 0.1, len(degrade_range))
    df.loc[degrade_range, "power_consumption"] += np.linspace(0, 300, len(degrade_range))
    df.loc[degrade_range, "current"] += np.linspace(0, 1.5, len(degrade_range))

    # ================================
    # CRITICAL FAILURE
    # ================================
    fail_range = range(221, NUM_SAMPLES)

    df.loc[fail_range, "process_temperature"] += 25
    df.loc[fail_range, "vibration"] += 0.3

    df.loc[fail_range, "rpm"] *= np.random.uniform(0.6, 0.8, len(fail_range))
    df.loc[fail_range, "torque"] += 35

    df.loc[fail_range, "current"] += 3
    df.loc[fail_range, "power_consumption"] += 800

    # More erratic idle behavior
    df.loc[fail_range, "idle_duration"] += np.random.uniform(2, 5, len(fail_range))

    # ================================
    # COLUMN ORDER FIX (CRITICAL)
    # ================================
    df = df[[
        "timestamp",
        "machine_id",
        "process_temperature",
        "air_temperature",
        "vibration",
        "torque",
        "rpm",
        "current",
        "operating_hours",
        "time_since_last_maintenance",
        "last_maintenance_Type",
        "idle_duration",
        "power_consumption"
    ]]

    # Save
    save_path = f"{DATA_DIR}/{FILE_NAME}"
    df.to_csv(save_path, index=False)

    print(f"\nGenerated: {FILE_NAME}")
    print(f"Path: {save_path}")
    print(f"Total Rows: {len(df)}")
    print("Structure:")
    print("  0–150   → Healthy")
    print(" 151–220  → Degrading")
    print(" 221–300  → Critical")

if __name__ == "__main__":
    generate_unlabeled_test_data()