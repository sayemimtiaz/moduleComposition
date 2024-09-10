import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
scratch_chart_file_path = 'scratch_chart.csv'
data = pd.read_csv(scratch_chart_file_path)

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot the Avg. Accuracy on the primary y-axis
ax1.plot(data['Data'], data['Avg. Accuracy'], color='blue', marker='o', linestyle='-', label='Avg. Accuracy')
ax1.set_xlabel('Data Sampling Rate', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color='black', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black')

# Create a secondary y-axis for Avg. Time
ax2 = ax1.twinx()
ax2.plot(data['Data'], data['Avg. Time (sec)'], color='red', marker='x', linestyle='-', label='Avg. Time (sec)')
ax2.set_ylabel('Time (sec)', color='black', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='black')

# Add title
# plt.title("Accuracy and Time vs Data Sampling Rate", fontsize=14, fontweight='bold')

# Add a legend at the top left
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=10)

# Remove extra white space and save as PDF
plt.tight_layout()
plt.savefig("line_chart_dual_axis.pdf", format='pdf')

# Display the plot
plt.show()
