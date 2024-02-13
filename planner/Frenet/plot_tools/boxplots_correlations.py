"""Create beautiful correleation boxplots."""

import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import argparse
import os

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_BLUE_TRANS20 = (0 / 255, 101 / 255, 189 / 255, 0.2)
TUM_BLUE_TRANS50 = (0 / 255, 101 / 255, 189 / 255, 0.5)
TUM_DARKBLUE = (0 / 255, 82 / 255, 147 / 255)
TUM_LIGHTBLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_ORANGE = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
PLOT_LIST = [
    "bayes<->equality",
    "bayes<->maximin",
    "bayes<->responsibility",
    "equality<->maximin",
    "equality<->responsibility",
    "maximin<->responsibility",
]

# Create dictionary of keyword aruments to pass to plt.boxplot
red_dict = {
    'patch_artist': True,
    'boxprops': {"color": TUM_BLUE, "facecolor": TUM_BLUE_TRANS50},
    'capprops': {"color": TUM_BLUE},
    'flierprops': {"color": TUM_BLUE_TRANS20, "markeredgecolor": TUM_BLUE_TRANS20},
    'medianprops': {"color": TUM_DARKBLUE},
    'whiskerprops': {"color": TUM_BLUE},
}


def colormap(val):
    """Different colors for positive and negative values.

    Args:
        val ([type]): [description]

    Returns:
        [type]: [description]
    """
    if val >= 0:
        return (0 / 255, 101 / 255, 189 / 255, val)

    if val < 0:
        return (227 / 255, 114 / 255, 34 / 255, -val)


cdict = {
    'red': [[0.0, 0.0, 0.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
    'green': [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 1.0, 1.0], [1.0, 1.0, 1.0]],
    'blue': [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]],
}

newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

top = plt.cm.get_cmap('Oranges_r', 128)
bottom = plt.cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')


def box_plot(data, edge_color, fill_color):
    """Generate a boxplot.

    Args:
        data ([type]): [description]
        edge_color ([type]): [description]
        fill_color ([type]): [description]

    Returns:
        [type]: [description]
    """
    bp = ax.boxplot(data, **red_dict)
    # print(bp.keys())

    # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    #     plt.setp(bp[element], color=edge_color)

    # for patch in bp['boxes']:
    #     patch.set(facecolor=fill_color)

    return bp


def plot_and_save(plt_data, title):
    """Plot and save.

    Args:
        plt_data (_type_): _description_
        title (_type_): _description_
    """
    _, ax1 = plt.subplots()
    _ = box_plot(plt_data.values(), TUM_DARKBLUE, TUM_LIGHTBLUE)
    ax1.set_xticklabels(plt_data.keys(), rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.resultdir, title + ".pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultdir", type=str, default="./planner/Frenet/results")
    args = parser.parse_args()

    with open(os.path.join(args.resultdir, 'corr_dict.json')) as json_file:
        corr_data = json.load(json_file)

    with open(os.path.join(args.resultdir, 'long_dict.json')) as json_file:
        long_data = json.load(json_file)

    with open(os.path.join(args.resultdir, 'lat_dict.json')) as json_file:
        lat_data = json.load(json_file)

    key_list = [
        "bayes",
        "equality",
        "maximin",
        "responsibility",
        "ego",
        "velocity",
        "dist_to_global_path",
    ]
    mean_dict = {key: np.mean(corr_data[key]) for key in corr_data}

    corr_mat = np.zeros((len(key_list), len(key_list)))
    for id1, key1 in enumerate(key_list):
        for id2, key2 in enumerate(key_list):
            if id1 == id2:
                corr_mat[id1, id2] = 1

            else:
                for keypair in mean_dict:
                    if key1 in keypair and key2 in keypair:
                        corr_mat[id1, id2] = mean_dict[keypair]

    fig = plt.figure("Correlation Matrix", figsize=(9, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        corr_mat,
        aspect='auto',
        cmap=newcmp,  # plt.cm.RdYlBu,
        interpolation='nearest',
    )

    ax.set_xticks(np.arange(len(key_list)))
    ax.set_yticks(np.arange(len(key_list)))
    ax.set_xticklabels(key_list)
    ax.set_yticklabels(key_list)

    cbar = fig.colorbar(im)
    im.set_clim(-1, 1)
    plt.savefig(os.path.join(args.resultdir, "correlations.pdf"))

    # corr_data = {key: corr_data[key] for key in corr_data if key in PLOT_LIST}
    # long_data = {key: long_data[key] for key in long_data if key in PLOT_LIST}
    # lat_data = {key: lat_data[key] for key in lat_data if key in PLOT_LIST}

    plot_and_save(corr_data, "correlations_boxplot")
    plot_and_save(long_data, "long_boxplot")
    plot_and_save(lat_data, "lat_boxplot")

    print("Done.")
