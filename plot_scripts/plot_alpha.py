import matplotlib.pyplot as plt

def plot_alphas(log_file, target_combination):
    """
    Read the log file and plot the alpha values for the specified combination.
    
    Args:
        log_file (str): Path to the simulation log file
        target_combination (str): The combination to plot (e.g., "PreferHBM + NoMigration")
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    alphas = []
    in_target_combination = False
    in_alphas_section = False
    
    for line in lines:
        line = line.strip()
        # Check for the target combination
        if line.startswith("Combination:"):
            current_combination = line.split("Combination:")[1].strip()
            in_target_combination = (current_combination == target_combination)
            in_alphas_section = False  # Reset alpha section flag
        # Start collecting alphas after "Alphas:"
        elif in_target_combination and line == "Alphas:":
            in_alphas_section = True
        # Stop at the separator
        elif in_alphas_section and line == "-" * 50:
            break
        # Collect alpha values
        elif in_alphas_section:
            try:
                alpha = float(line)
                alphas.append(alpha)
            except ValueError:
                continue  # Skip lines that can't be converted to float
    
    if not alphas:
        print(f"No alpha values found for combination: {target_combination}")
        return
    
    # Plot the alpha values
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, label=f"Alpha ({target_combination})")
    plt.xlabel("Step")
    plt.ylabel("Alpha")
    plt.title(f"Alpha Variation Over Simulation Steps\nCombination: {target_combination}")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # log_file_path = "./skip0/1024_2048_0.5B_4GB_TokenLevelBestRatioInit_20250317_175416.txt"  # Adjust to your log file path
    # combination_to_plot = "PreferHBM + NoMigration"  # Adjust to the desired combination
    log_file_path = "./skip0/1024_16384_0.5B_4GB_TokenLevelBestRatioInit_20250318_104329.txt"  # Adjust to your log file path
    combination_to_plot = "AlphaLayersDistribution + AlphaMigration"
    plot_alphas(log_file_path, combination_to_plot)