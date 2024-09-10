import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV data
csv_file_path = 'rq2.csv'
data = pd.read_csv(csv_file_path)

# Select the columns containing accuracy scores
accuracy_data = data[['Scratch', 'MU', 'MS']]

# Colors for the boxes
colors = ['lightcoral', 'lightsalmon', 'lightpink']

# Create a box plot with customized colors
plt.figure(figsize=(6, 6))
box = plt.boxplot(accuracy_data.values, patch_artist=True, labels=accuracy_data.columns, medianprops=dict(color='black'))

# Set colors for each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Add title and labels with increased font size and bold text
plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Reuse Mode", fontsize=14, fontweight='bold')

# Add median values as text annotations above the median line
medians = [np.median(accuracy_data[col]) for col in accuracy_data.columns]
for i, median in enumerate(medians):
    plt.text(i + 1, median + 1, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

# Adjust layout to remove whitespace
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("rq2.pdf", format='pdf')

# Display the plot
plt.show()
