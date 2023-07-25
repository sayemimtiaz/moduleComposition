import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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
                        backgroundcolor=fillcolor, color=color,
                        fontsize=12)


fig, axs = plt.subplots(1, 2, figsize=(24, 10), squeeze=False, sharey=True)

for i, model_suffix in enumerate(["1", "4"]):
    if model_suffix == '1':
        edgecolor = 'blue'
        fillcolor = 'lightblue'
    else:
        edgecolor = 'peru'
        fillcolor = 'peachpuff'
    fileName = os.path.join(base_path, "result", "combined_model" + model_suffix + ".csv")

    df = pd.read_csv(fileName)

    df.rename(columns={
        'DeepCompose Accuracy': 'DeepCompose',
        'Static Composition Accuracy': 'Voting',
        'Module Stacking Accuracy': 'Stacking',
        'Train-from-scratch Accuracy': 'Scratch'
    }, inplace=True)

    bp, props = df.boxplot(column=['DeepCompose', 'Voting', 'Stacking', 'Scratch'],
                           return_type='both',
                           ax=axs[0, i],
                           boxprops=dict(linestyle='-', linewidth=1, color=edgecolor, facecolor=fillcolor),
                           flierprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                           medianprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                           whiskerprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                           capprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                           showfliers=False, grid=True, rot=0, vert=True, meanline=True,
                           patch_artist=True
                           )
    add_values(props, axs[0, i], color=edgecolor)

    # Remove top and right spines
    axs[0, i].spines['top'].set_visible(False)
    axs[0, i].spines['right'].set_visible(False)
    if model_suffix=='4':
        axs[0, i].spines['left'].set_visible(False)

    axs[0, i].tick_params(labelsize=16)

    axs[0, i].set_title(f'Model '+model_suffix, fontweight='bold', fontsize=18, y=1.05)


    # axs[0, i].set_xlabel('Reuse mode', fontweight='bold', fontsize=18, labelpad=15)
    # axs[0, i].set_ylabel('Accuracy', fontweight='bold', fontsize=18, labelpad=15)

# Set xlabel and ylabel for the entire figure
fig.text(0.5, 0.02, 'Reuse mode', fontweight='bold', fontsize=18, ha='center')
fig.text(0.08, 0.5, 'Accuracy', fontweight='bold', fontsize=18, va='center', rotation='vertical')

plt.savefig('figure/boxplotModel' + '.pdf')
plt.show()
