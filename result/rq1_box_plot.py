import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mode = 'accuracy'  # accuracy or time


def series_values_as_dict(series_object):
    tmp = series_object.to_dict().values()
    return [y for y in tmp][0]


def add_values(bp, ax, color="blue"):
    global mode
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
                if mode == 'accuracy':
                    s = 0.00
                    x_line_center = x_l + 0.35
                    y_line_center = y
                    fillcolor = 'white'
                else:
                    s = 0.0
                    x_line_center = x_l + 0.2
                    y_line_center = y
                    fillcolor = 'white'
            ax.text(x_line_center - s, y_line_center,  # Position
                    '%.1f' % y,  # Value (3f = 3 decimal float)
                    verticalalignment='center',  # Centered vertically with line
                    backgroundcolor=fillcolor, color=color,
                    fontsize=12)


if mode == 'accuracy':
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False, sharey=True)
else:
    fig, axs = plt.subplots(1, 1, figsize=(9, 10), squeeze=False, sharey=True)

fileName = os.path.join(base_path, "result", "rq1" + ".csv")

df = pd.read_csv(fileName)

if mode == 'accuracy':
    edgecolor = 'darkgoldenrod'
    fillcolor = 'cornsilk'
else:
    edgecolor = 'darkviolet'
    fillcolor = 'thistle'


if mode == 'accuracy':
    cols = ['Scratch', 'Voting-only', 'SC+Voting', 'Update', 'Stacking']
    df.rename(columns={
        'Static': 'Voting-only',
        'SC': 'SC+Voting',
        'MU': 'Update',
        'MS': 'Stacking',
        'Scratch': 'Scratch'
    }, inplace=True)
else:
    cols = ['DeepCompose', 'Scratch']
    df.rename(columns={
        'DeepCompose Training Time': 'DeepCompose',
        # 'Module Stacking Training Time': 'Stacking',
        'Train-from-scratch Training Time': 'Scratch'
    }, inplace=True)
    df['DeepCompose'] = (df['DeepCompose'] / 60)
    df['Scratch'] = (df['Scratch'] / 60)

df = df[cols]
print(df.describe())

bp, props = df.boxplot(column=cols,
                       return_type='both',
                       ax=axs[0, 0],
                       boxprops=dict(linestyle='-', linewidth=1, color=edgecolor, facecolor=fillcolor),
                       flierprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                       medianprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                       whiskerprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                       capprops=dict(linestyle='-', linewidth=1, color=edgecolor),
                       meanprops=dict(color='orange', marker='D'),
                       showfliers=False, grid=True, rot=0, vert=True, meanline=False,
                       patch_artist=True,
                       showmeans=False
                       )
add_values(props, axs[0, 0], color=edgecolor)

# Remove top and right spines
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
# if model_suffix == '4':
#     axs[0, i].spines['left'].set_visible(False)

axs[0, 0].tick_params(labelsize=16)

# tms = model_suffix
# if model_suffix == '4':
#     tms = '2'
# axs[0, i].set_title(f'Model ' + tms, fontweight='bold', fontsize=18, y=1.05)

# axs[0, i].set_xlabel('Reuse mode', fontweight='bold', fontsize=18, labelpad=15)
# axs[0, i].set_ylabel('Accuracy', fontweight='bold', fontsize=18, labelpad=15)

# Set xlabel and ylabel for the entire figure
fig.text(0.5, 0.02, 'Reuse mode', fontweight='bold', fontsize=18, ha='center')
if mode == 'accuracy':
    yl = 'Accuracy'
    ylx = 0.04
    yly = 0.5
else:
    yl = 'Time to converge (in min)'
    ylx = 0.01
    yly = 0.5
fig.text(ylx, yly, yl, fontweight='bold', fontsize=18, va='center', rotation='vertical')

plt.savefig('figure/boxplotModel' + mode + '.pdf')
plt.show()
