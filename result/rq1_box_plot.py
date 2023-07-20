import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model_suffix = '1'

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fileName = os.path.join(base_path, "result", "combined_model" + model_suffix + ".csv")


def series_values_as_dict(series_object):
    tmp = series_object.to_dict().values()
    return [y for y in tmp][0]


def add_values(bp, ax, color="blue"):
    """ This actually adds the numbers to the various points of the boxplots"""
    for element in ['medians']:
        # for element in ['whiskers', 'medians', 'caps']:
        for line in bp[element]:
            # Get the position of the element. y is the label you want
            (x_l, y), (x_r, _) = line.get_xydata()
            # Make sure datapoints exist
            # (I've been working with intervals, should not be problem for this case)
            if not np.isnan(y):
                x_line_center = x_l + (x_r - x_l) / 2
                y_line_center = y  # Since it's a line and it's horisontal
                # overlay the value:  on the line, from center to right
                ax.text(x_line_center, y_line_center,  # Position
                        '%.1f' % y,  # Value (3f = 3 decimal float)
                        verticalalignment='center',  # Centered vertically with line
                        backgroundcolor="lightblue", color=color,
                        fontsize=12)


df = pd.read_csv(fileName)

df.rename(columns={
    'DeepCompose Accuracy': 'DeepCompose',
    'Static Composition Accuracy': 'Static',
    'Module Stacking Accuracy': 'Stacking',
    'Train-from-scratch Accuracy': 'Scratch'
}, inplace=True)

print(df.head())

fig, ax = plt.subplots(1, figsize=(12, 10))
color = 'blue'
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']

bp, props = df.boxplot(column=['DeepCompose', 'Static', 'Stacking', 'Scratch'],
                  return_type='both',
                  boxprops=dict(linestyle='-', linewidth=1, color=color, facecolor='lightblue'),
                  flierprops=dict(linestyle='-', linewidth=1, color=color),
                  medianprops=dict(linestyle='-', linewidth=1, color=color),
                  whiskerprops=dict(linestyle='-', linewidth=1, color=color),
                  capprops=dict(linestyle='-', linewidth=1, color=color),
                  showfliers=False, grid=True, rot=0, vert=True, meanline=True,
                  patch_artist=True,
                  )
add_values(props, ax, color=color)

# for item in ['boxes']:
#     for patch, color in zip(props[item], colors):
#         patch.set_color(color)


# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel('Reuse mode', fontweight='bold', fontsize=18, labelpad=15)
plt.ylabel('Accuracy', fontweight='bold', fontsize=18, labelpad=15)

plt.savefig('figure/boxplotModel'+model_suffix+'.pdf')
plt.show()

