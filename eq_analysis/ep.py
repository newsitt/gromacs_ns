import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse

def load_xvg_data(file_path):
    """
    Load .xvg data while ignoring metadata lines (comments starting with "@" or "#").
    """
    try:
        data = np.loadtxt(file_path, comments=["@", "#"])
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_potential_energy(time, potential_energy):
    """
    Compute statistical metrics for potential energy.
    """
    mean_potential_energy = np.mean(potential_energy)
    std_potential_energy = np.std(potential_energy)
    cv_potential_energy = (std_potential_energy / mean_potential_energy) * 100  # Coefficient of Variation (%)
    standard_error_potential_energy = std_potential_energy / np.sqrt(len(potential_energy))

    return mean_potential_energy, std_potential_energy, cv_potential_energy, standard_error_potential_energy

def check_equilibration(std_potential_energy, cv_potential_energy, standard_error_potential_energy):
    """
    Check if the system is equilibrated based on predefined thresholds.
    """
    equilibration_criteria = {
        "std_threshold": 500,   # Standard deviation should be < 500 kJ/mol
        "cv_threshold": 1.0,    # Coefficient of Variation (%) should be < 1%
        "se_threshold": 5.0     # Standard Error should be < 5 kJ/mol
    }

    is_equilibrated = (
        std_potential_energy < equilibration_criteria["std_threshold"] and
        cv_potential_energy < equilibration_criteria["cv_threshold"] and
        standard_error_potential_energy < equilibration_criteria["se_threshold"]
    )

    return is_equilibrated

def plot_potential_energy(time, potential_energy, mean_potential_energy):
    """
    Plot potential energy over time with a trend line.
    """
    slope, intercept, _, _, _ = linregress(time, potential_energy)
    trend_line = slope * time + intercept

    plt.figure(figsize=(8, 5))
    plt.plot(time, potential_energy, label="Potential Energy", color="b", linewidth=1, alpha=0.7)
    plt.axhline(mean_potential_energy, color="r", linestyle="--", label="Mean Energy")
    plt.plot(time, trend_line, color="g", linestyle="-.", linewidth=2, label="Trend Line (Linear Fit)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.title("Potential Energy Over Time with Trend Line")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze potential energy from an .xvg file.")
    parser.add_argument("file_path", type=str, help="Path to the .xvg file")
    args = parser.parse_args()

    data = load_xvg_data(args.file_path)

    if data is not None:
        time, potential_energy = data[:, 0], data[:, 1]

        # Compute statistics
        mean_pe, std_pe, cv_pe, se_pe = analyze_potential_energy(time, potential_energy)

        # Check equilibration
        is_equilibrated = check_equilibration(std_pe, cv_pe, se_pe)
        equilibration_message = "✅ The system is equilibrated." if is_equilibrated else "❌ The system is NOT equilibrated yet."

        # Summary DataFrame
        summary_df = pd.DataFrame({
            "Metric": ["Mean Potential Energy", "Standard Deviation", "Coefficient of Variation (%)", "Standard Error"],
            "Value": [mean_pe, std_pe, cv_pe, se_pe]
        })

        # Print summary and equilibration status
        print("\n🔍 Potential Energy Analysis Summary:")
        print(summary_df.to_string(index=False))
        print("\n🔍 Equilibration Status:")
        print(equilibration_message)

        # Plot the potential energy data
        plot_potential_energy(time, potential_energy, mean_pe)

if __name__ == "__main__":
    main()
