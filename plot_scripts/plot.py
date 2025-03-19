import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_simulation_data(data):
    """
    Plots a grouped bar chart of total simulation times for different data migration and placement strategies.
    Displays the average alpha rate for each migration strategy beneath the x-axis labels.
    Shows the exact simulation time on top of each bar.

    Parameters:
    data (list of dict): Each dictionary should contain 'placement', 'migration', 'time', and 'alpha'.
                         Example: [{'placement': 'PreferHBM', 'migration': 'NoMigration', 'time': 23.2145, 'alpha': 0.992730}, ...]
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort the DataFrame for consistent ordering
    df = df.sort_values(by=['migration', 'placement'])
    
    # Calculate average alpha per migration strategy
    avg_alpha = df.groupby('migration')['alpha'].mean().to_dict()
    
    # Create new x-axis labels with average alpha
    unique_migrations = df['migration'].unique()
    new_labels = [f"{mig}\nAlpha: {avg_alpha[mig]:.6f}" for mig in unique_migrations]
    
    # Create the figure with a suitable size
    plt.figure(figsize=(14, 8))
    
    # Generate a grouped bar plot
    ax = sns.barplot(data=df, x='migration', y='time', hue='placement', palette='Set2')
    
    # Customize the y-axis range and ticks to match your data
    plt.ylim(22.0, 23.5)
    plt.yticks([22.0 + 0.1 * i for i in range(16)])
    
    # Set the new x-axis labels with alpha values
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    
    # Add titles and labels
    plt.title('Total Simulation Time for Data Migration and Placement Strategies')
    plt.xlabel('Data Migration Strategy with Average Alpha')
    plt.ylabel('Total Time (seconds)')
    
    # Position the legend outside the plot
    plt.legend(title='Data Placement', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add exact time values on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Ensure text is only added to visible bars
            ax.text(p.get_x() + p.get_width() / 2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Display the plot
    plt.show()

# Example usage with your sample data
if __name__ == "__main__":
    # prefer_hbm_data = [
    #     {'placement': 'PreferHBM', 'migration': 'NoMigration', 'time': 23.2145, 'alpha': 0.992730},
    #     {'placement': 'PreferHBM', 'migration': 'PriorMigration', 'time': 23.2174, 'alpha': 0.992803},
    #     {'placement': 'PreferHBM', 'migration': 'SkippedTokensMigration', 'time': 23.2188, 'alpha': 0.992855},
    #     {'placement': 'PreferHBM', 'migration': 'PastWindowMigration', 'time': 23.1888, 'alpha': 0.991934},
    #     {'placement': 'PreferHBM', 'migration': 'LookAheadMigration', 'time': 23.3428, 'alpha': 0.992823},
    #     {'placement': 'PreferHBM', 'migration': 'LookAheadBatchMigration', 'time': 23.2145, 'alpha': 0.992730},
    #     {'placement': 'PreferHBM', 'migration': 'AlphaMigration', 'time': 22.7326, 'alpha': 0.978327}
    # ]

    # split_token_data = [
    #     {'placement': 'SplitToken', 'migration': 'NoMigration', 'time': 23.1925, 'alpha': 0.992057},
    #     {'placement': 'SplitToken', 'migration': 'PriorMigration', 'time': 23.1948, 'alpha': 0.992115},
    #     {'placement': 'SplitToken', 'migration': 'SkippedTokensMigration', 'time': 23.1958, 'alpha': 0.992153},
    #     {'placement': 'SplitToken', 'migration': 'PastWindowMigration', 'time': 23.1645, 'alpha': 0.991196},
    #     {'placement': 'SplitToken', 'migration': 'LookAheadMigration', 'time': 23.3428, 'alpha': 0.992823},
    #     {'placement': 'SplitToken', 'migration': 'LookAheadBatchMigration', 'time': 23.2145, 'alpha': 0.992731},
    #     {'placement': 'SplitToken', 'migration': 'AlphaMigration', 'time': 22.7240, 'alpha': 0.978082}
    # ]

    # batch_ratio_data = [
    #     {'placement': 'BatchRatio', 'migration': 'NoMigration', 'time': 23.1928, 'alpha': 0.992067},
    #     {'placement': 'BatchRatio', 'migration': 'PriorMigration', 'time': 23.1950, 'alpha': 0.992120},
    #     {'placement': 'BatchRatio', 'migration': 'SkippedTokensMigration', 'time': 23.1962, 'alpha': 0.992164},
    #     {'placement': 'BatchRatio', 'migration': 'PastWindowMigration', 'time': 23.1645, 'alpha': 0.991195},
    #     {'placement': 'BatchRatio', 'migration': 'LookAheadMigration', 'time': 23.3428, 'alpha': 0.992823},
    #     {'placement': 'BatchRatio', 'migration': 'LookAheadBatchMigration', 'time': 23.2145, 'alpha': 0.992730},
    #     {'placement': 'BatchRatio', 'migration': 'AlphaMigration', 'time': 22.7239, 'alpha': 0.978077}
    # ]

    # look_ahead_batch_data = [
    #     {'placement': 'LookAheadBatch', 'migration': 'NoMigration', 'time': 23.2145, 'alpha': 0.992730},
    #     {'placement': 'LookAheadBatch', 'migration': 'PriorMigration', 'time': 23.2174, 'alpha': 0.992803},
    #     {'placement': 'LookAheadBatch', 'migration': 'SkippedTokensMigration', 'time': 23.2188, 'alpha': 0.992855},
    #     {'placement': 'LookAheadBatch', 'migration': 'PastWindowMigration', 'time': 23.1888, 'alpha': 0.991934},
    #     {'placement': 'LookAheadBatch', 'migration': 'LookAheadMigration', 'time': 23.3428, 'alpha': 0.992823},
    #     {'placement': 'LookAheadBatch', 'migration': 'LookAheadBatchMigration', 'time': 23.2145, 'alpha': 0.992730},
    #     {'placement': 'LookAheadBatch', 'migration': 'AlphaMigration', 'time': 22.7326, 'alpha': 0.978327}
    # ]

    # layer_importance_data = [
    #     {'placement': 'LayerImportance', 'migration': 'NoMigration', 'time': 22.6824, 'alpha': 0.976919},
    #     {'placement': 'LayerImportance', 'migration': 'PriorMigration', 'time': 22.6824, 'alpha': 0.976919},
    #     {'placement': 'LayerImportance', 'migration': 'SkippedTokensMigration', 'time': 22.6824, 'alpha': 0.976919},
    #     {'placement': 'LayerImportance', 'migration': 'PastWindowMigration', 'time': 22.6627, 'alpha': 0.976327},
    #     {'placement': 'LayerImportance', 'migration': 'LookAheadMigration', 'time': 23.3428, 'alpha': 0.992823},
    #     {'placement': 'LayerImportance', 'migration': 'LookAheadBatchMigration', 'time': 23.2145, 'alpha': 0.992731},
    #     {'placement': 'LayerImportance', 'migration': 'AlphaMigration', 'time': 22.6625, 'alpha': 0.976322}
    # ]

    # 0.8 model weight
    prefer_hbm_data = [
        {'placement': 'PreferHBM', 'migration': 'NoMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'PreferHBM', 'migration': 'PriorMigration', 'time': 23.1647, 'alpha': 0.872375},
        {'placement': 'PreferHBM', 'migration': 'SkippedTokensMigration', 'time': 23.1660, 'alpha': 0.872426},
        {'placement': 'PreferHBM', 'migration': 'PastWindowMigration', 'time': 23.1359, 'alpha': 0.871500},
        {'placement': 'PreferHBM', 'migration': 'LookAheadMigration', 'time': 23.5634, 'alpha': 0.872395},
        {'placement': 'PreferHBM', 'migration': 'LookAheadBatchMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'PreferHBM', 'migration': 'AlphaMigration', 'time': 22.6775, 'alpha': 0.857830}
    ]
    
    split_token_data = [
        {'placement': 'SplitToken', 'migration': 'NoMigration', 'time': 23.1397, 'alpha': 0.871629},
        {'placement': 'SplitToken', 'migration': 'PriorMigration', 'time': 23.1421, 'alpha': 0.871686},
        {'placement': 'SplitToken', 'migration': 'SkippedTokensMigration', 'time': 23.1431, 'alpha': 0.871725},
        {'placement': 'SplitToken', 'migration': 'PastWindowMigration', 'time': 23.1116, 'alpha': 0.870762},
        {'placement': 'SplitToken', 'migration': 'LookAheadMigration', 'time': 23.5634, 'alpha': 0.872395},
        {'placement': 'SplitToken', 'migration': 'LookAheadBatchMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'SplitToken', 'migration': 'AlphaMigration', 'time': 22.6692, 'alpha': 0.857591}
    ]

    batch_ratio_data = [
        {'placement': 'BatchRatio', 'migration': 'NoMigration', 'time': 23.1399, 'alpha': 0.871633},
        {'placement': 'BatchRatio', 'migration': 'PriorMigration', 'time': 23.1421, 'alpha': 0.871687},
        {'placement': 'BatchRatio', 'migration': 'SkippedTokensMigration', 'time': 23.1432, 'alpha': 0.871730},
        {'placement': 'BatchRatio', 'migration': 'PastWindowMigration', 'time': 23.1116, 'alpha': 0.870761},
        {'placement': 'BatchRatio', 'migration': 'LookAheadMigration', 'time': 23.5634, 'alpha': 0.872395},
        {'placement': 'BatchRatio', 'migration': 'LookAheadBatchMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'BatchRatio', 'migration': 'AlphaMigration', 'time': 22.6697, 'alpha': 0.857607}
    ]
    
    look_ahead_batch_data = [
        {'placement': 'LookAheadBatch', 'migration': 'NoMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'LookAheadBatch', 'migration': 'PriorMigration', 'time': 23.1647, 'alpha': 0.872375},
        {'placement': 'LookAheadBatch', 'migration': 'SkippedTokensMigration', 'time': 23.1660, 'alpha': 0.872426},
        {'placement': 'LookAheadBatch', 'migration': 'PastWindowMigration', 'time': 23.1359, 'alpha': 0.871500},
        {'placement': 'LookAheadBatch', 'migration': 'LookAheadMigration', 'time': 23.5634, 'alpha': 0.872395},
        {'placement': 'LookAheadBatch', 'migration': 'LookAheadBatchMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'LookAheadBatch', 'migration': 'AlphaMigration', 'time': 22.6775, 'alpha': 0.857830}
    ]
    
    layer_importance_data = [
        {'placement': 'LayerImportance', 'migration': 'NoMigration', 'time': 22.6295, 'alpha': 0.856485},
        {'placement': 'LayerImportance', 'migration': 'PriorMigration', 'time': 22.6295, 'alpha': 0.856485},
        {'placement': 'LayerImportance', 'migration': 'SkippedTokensMigration', 'time': 22.6295, 'alpha': 0.856485},
        {'placement': 'LayerImportance', 'migration': 'PastWindowMigration', 'time': 22.6098, 'alpha': 0.855893},
        {'placement': 'LayerImportance', 'migration': 'LookAheadMigration', 'time': 23.5634, 'alpha': 0.872395},
        {'placement': 'LayerImportance', 'migration': 'LookAheadBatchMigration', 'time': 23.1617, 'alpha': 0.872302},
        {'placement': 'LayerImportance', 'migration': 'AlphaMigration', 'time': 22.6096, 'alpha': 0.855888}
    ]
    plot_simulation_data(prefer_hbm_data)
    plot_simulation_data(split_token_data)
    plot_simulation_data(batch_ratio_data)
    plot_simulation_data(look_ahead_batch_data)
    plot_simulation_data(layer_importance_data)