# Verification.py
# Independent test harness to verify metrics from simulator_output.npz
# against pre-defined target values.

import numpy as np
import sys
import os

# --- TARGETS ---
# These values represent the expected output from a correctly tuned and executed simulation.
TARGETS = {
    "halo": {
        "r_flat": 67.632, "v_flat": 0.300, "M_dyn": 6.08684,
        "A_fit": 28.0959, "M_lens": 7.02396, "ratio": 1.15396,
        "rtol": 0.08, "atol": 1e-3
    },
    "void": {
        "boundary_fraction": 0.0, "inward_bias": 0.468211,
        "rtol": 0.10, "atol": 1e-3
    },
    "aniso": {
        "A0": 1.7654, "A1c": -0.95309, "A1s": 0.0115941,
        "A2c": 0.04375, "A2s": 0.0145251, "P_k3": 0.179434,
        "rtol": 0.12, "atol": 1e-3
    },
}

def verify_metrics(loaded_metrics, targets):
    """
    Compares loaded metrics against target values and reports results.
    """
    all_pass = True
    print("\n--- Verification Report ---")

    for block_name, block_targets in targets.items():
        print(f"\n[{block_name.upper()} BLOCK]")
        block_pass = True

        # Extract tolerances
        rtol = block_targets['rtol']
        atol = block_targets['atol']

        # Check all metrics in the block
        for metric_name, target_value in block_targets.items():
            if metric_name in ['rtol', 'atol']:
                continue

            # Metrics are stored with 'metric_' prefix in the NPZ file (e.g., 'metric_ratio')
            npz_key = f'metric_{metric_name}'

            if npz_key not in loaded_metrics:
                print(f"  [MISSING] {metric_name}: Key '{npz_key}' not found in simulator output.")
                block_pass = False
                continue

            # Use .item() to extract the scalar float from the numpy array/scalar
            actual_value = loaded_metrics[npz_key].item()

            # Comparison using numpy's tolerance check
            # For target_value == 0.0, rtol is not meaningful, so we rely on atol.
            if target_value == 0.0:
                is_pass = np.isclose(actual_value, target_value, atol=atol)
                tolerance_info = f"atol={atol}"
            else:
                is_pass = np.isclose(actual_value, target_value, rtol=rtol, atol=atol)
                tolerance_info = f"rtol={rtol}, atol={atol}"

            status = "PASS" if is_pass else "FAIL"

            # Print result
            print(f"  [{status:^4s}] {metric_name:<20s} Target: {target_value: <10.6g} Actual: {actual_value: <10.6g} ({tolerance_info})")

            if not is_pass:
                block_pass = False
                all_pass = False

        print(f"[{block_name.upper()} BLOCK] {'PASSED' if block_pass else 'FAILED'}")

    print("\n--------------------------")
    print(f"OVERALL VERIFICATION: {'SUCCESS' if all_pass else 'FAILURE'}")
    print("--------------------------")
    return all_pass

if __name__ == "__main__":
    npz_file = 'simulator_output.npz'

    if not os.path.exists(npz_file):
        print(f"Error: Required file '{npz_file}' not found.")
        print("Please ensure AutoTune.py and Simulator3.py have been run successfully.")
        sys.exit(1)

    # Load the compressed data
    try:
        data = np.load(npz_file, allow_pickle=True)
        print(f"Successfully loaded data from {npz_file}. Keys: {list(data.keys())}")
        verify_metrics(data, TARGETS)

    except Exception as e:
        print(f"An error occurred while loading or verifying the data: {e}")
        sys.exit(1)