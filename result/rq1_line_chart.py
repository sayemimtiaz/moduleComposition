import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('mod_stat.csv')

# Plotting
plt.figure(figsize=(12, 8))

# List of columns to plot (excluding #Module)
columns_to_plot = [col for col in data.columns if col != '#Module']

# Define colors for each column
colors = {
    'Stacking': '#9ed866',  # Rich pastel orange
    'Update': '#f7c85f',    # Rich pastel blue
    'Scratch': '#ca472f',   # Dark green
    'Voting-only': '#0e7997', # Rich pastel red
    'SC+Voting': '#6f4e7c'  # Rich pastel cyan
}

# Plot each column's data
for column in columns_to_plot:
    color = colors.get(column, '#B0B0B0')  # Default medium gray if not specified
    plt.plot(data['#Module'], data[column], marker='o', linestyle='-', color=color, label=column)

# Customize the chart
plt.xlabel('Number of Modules', fontsize=12, fontweight='bold')
plt.ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(False)  # Remove gridlines
plt.xlim(data['#Module'].min(), data['#Module'].max())  # Adjust x-axis limit based on your data
plt.ylim(0, 100)  # Adjust y-axis limit based on your data

plt.tight_layout()

# Show plot
plt.show()
