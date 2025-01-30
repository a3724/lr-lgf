import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from matplotlib.colors import LinearSegmentedColormap
from tueplots.constants.color import rgb
from matplotlib import ticker
from tueplots import bundles
import os


plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi":200})

import seaborn as sns

# Seaborn
sns.set_theme(style="white")

# TUEplots
plt.rcParams.update(bundles.neurips2023())
plt.rcParams.update({"figure.dpi": 150})
plt.rcParams.update({"figure.constrained_layout.w_pad": 0.07})
plt.rcParams['axes.labelpad'] = '2'
plt.rcParams['xtick.major.pad'] = '4'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

rw = LinearSegmentedColormap.from_list("rwg", colors=[(1, 1, 1), rgb.tue_red], N=1024)
colors = [rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve, rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve, rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve,rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve, rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve, rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve]

def load_data(data_dir, exp, methods, num_tasks):
    exp_dir = os.path.join(data_dir, exp)

    # Loop over methods
    res_dicts = dict()
    for m in methods:
        path = os.path.join(exp_dir, m)

        # Loop over seeds
        seeds = [f.path for f in os.scandir(path) if f.is_dir()]
        all_accuracies = []
        for s, s_path in enumerate(seeds):
            results = np.loadtxt(os.path.join(s_path, 'accuracies_matrix.csv'),
                                 delimiter=',')
            all_accuracies.append(results)
        res_dicts[m] = np.array(all_accuracies)

    return res_dicts

def plot_performance(results, num_tasks, title, colors, ax, left_plot=False):
    SEED = 6
    results = results[SEED]

    # Compute mean (ignoring zeros)
    mean = results.copy()
    mean[mean == 0] = np.nan
    mean = np.nanmean(mean, axis=1)

    # Loop over tasks
    for t in range(num_tasks):
        ax.plot(np.arange(t + 1, num_tasks + 1),
                results[t:, t],
                'o-',
                color=colors[1],
                alpha=0.9,
                zorder=1,
                markeredgewidth=0)
        ax.plot([t + 1],
                results[t, t],
                'X-',  # Using capital X for a thicker marker
                color=colors[-1],
                alpha=0.9,
                zorder=10,
                markersize=8,  # Increase marker size
                markeredgewidth=0)
    ax.plot(np.arange(1, num_tasks + 1),
            mean,
            'o-',
            color=colors[0],
            alpha=0.9,
            zorder=5)

    # Polish layout
    ax.grid(axis='x')
    ax.set_ylim(0.7, 1.01)
    ax.set_xlim(0.9, num_tasks + 0.1)
    ax.set_xticks(np.arange(1, num_tasks + 1))
    ax.tick_params('both', length=4, width=1, which='major')
    ax.set_xlabel(r'Task $t$')
    ax.set_title(title)
    if left_plot:
        ax.set_ylabel('Accuracy')
    if not left_plot:
        ax.tick_params('y', length=0, width=0, which='major')


def plot_performance_seeds(results,
                           num_tasks,
                           mean,
                           colors,
                           ax,
                           left_plot=False):
    # Loop over dict
    i = 0
    for m, res in results.items():
        # Loop over seeds
        num_seeds = len(res)
        task_specific_model = []
        average_perf = []
        for s in range(num_seeds):
            task_specific_model.append(np.diag(res[s]))
            mean = res[s].copy()
            mean[mean == 0] = np.nan
            mean = np.nanmean(mean, axis=1)
            average_perf.append(mean)
        average_perf_mean = np.mean(average_perf, axis=0)
        average_perf_std = np.std(average_perf, axis=0)
        task_spec_model_mean = np.mean(task_specific_model, axis=0)
        task_spec_model_std = np.std(task_specific_model, axis=0)

        # Labels
        if m == "lam0_Qall0_Qstrc00_c10":
            label = "No Regularization"
        elif m == "lam100000_Qall0_Qstrc00_c10":
            label = r"$\lambda\!>\!0$"
        elif m == "lam100000_Qall1e-06_Qstrc00_c10":
            label = r"$\lambda\!>\!0$, Scalar $Q$"
        elif m == "Q_str":
            label = r"$\lambda\!>\!0$, Structured $Q$"

        if left_plot:
            ax.plot(np.arange(1, num_tasks + 1),
                    average_perf_mean,
                    'o-',
                    color=colors[i],
                    alpha=0.9,
                    zorder=5,
                    label=label)
            ax.fill_between(np.arange(1, num_tasks + 1),
                            average_perf_mean - average_perf_std,
                            average_perf_mean + average_perf_std,
                            color=colors[i],
                            alpha=0.1)

        else:
            ax.plot(np.arange(1, num_tasks + 1),
                    task_spec_model_mean,
                    'o-',
                    color=colors[i],
                    alpha=0.9,
                    zorder=5,
                    label=label,
                    markeredgewidth=0)
            ax.fill_between(np.arange(1, num_tasks + 1),
                            task_spec_model_mean - task_spec_model_std,
                            task_spec_model_mean + task_spec_model_std,
                            color=colors[i],
                            alpha=0.1)

        i += 1

    # Polish layout
    ax.grid(axis='x')
    ax.set_ylim(0.7, 0.95)
    ax.set_xlim(0.9, num_tasks + 0.1)
    ax.set_xticks(np.arange(1, num_tasks + 1))
    ax.tick_params('both', length=4, width=1, which='major')
    ax.set_xlabel(r'Task $t$')
    if left_plot:
        ax.set_title("Average Performance Across Tasks")
        ax.set_ylabel('Accuracy')
    if not left_plot:
        ax.set_title("Performance on Current Task")
        ax.legend()
