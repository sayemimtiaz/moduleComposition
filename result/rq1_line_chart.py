import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('mod_stat.csv')

# Set a clean style, without internal gridlines
sns.set(style="white")

# Increase figure size for better visibility
# plt.figure(figsize=(14, 8))

# List of columns to plot (excluding #Module)
columns_to_plot = [col for col in data.columns if col != '#Module']

# Define colors for each column (using a visually appealing palette)
colors = {
    'MS': '#ff7f0e',     # Orange
    'MU': '#1f77b4',     # Blue
    'Scratch': '#2ca02c',  # Green
    'Static': '#d62728',  # Red
    'SC': '#9467bd'      # Purple
}

# Plot each column's data with markers and thicker lines for clarity
for column in columns_to_plot:
    color = colors.get(column, '#B0B0B0')  # Default medium gray if not specified
    plt.plot(data['#Module'], data[column],
             marker='o', linestyle='-', color=color,
             markersize=4, linewidth=1.5, label=column)

# Customize the chart
plt.xlabel('Number of Modules', fontsize=14, fontweight='bold')
plt.ylabel('Mean Accuracy', fontsize=14, fontweight='bold')

# Remove gridlines
plt.grid(False)

# Adjust axis limits for better spacing
plt.xlim(data['#Module'].min(), data['#Module'].max())
plt.ylim(0, 100)

# Style the legend
plt.legend(title='Reuse Mode', title_fontsize=13, fontsize=11, loc='best')

# Add thicker borders around the plot (spines)
plt.gca().spines['top'].set_linewidth(1)
plt.gca().spines['right'].set_linewidth(1)
plt.gca().spines['bottom'].set_linewidth(1)
plt.gca().spines['left'].set_linewidth(1)

# Tighten the layout to avoid overlaps
plt.tight_layout()
plt.savefig('rq1Line.pdf')
# Show plot
plt.show()
